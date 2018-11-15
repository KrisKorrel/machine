import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.gumbel import gumbel_softmax
from ..util.sparsemax import Sparsemax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionActivation(nn.Module):
    def __init__(self, sample_train='full', sample_infer='full', learn_temperature='no', initial_temperature=0.):
        super(AttentionActivation, self).__init__()
        self.sample_train = sample_train
        self.sample_infer = sample_infer

        if 'gumbel' in sample_train or sample_train == 'full_hard':
            self.learn_temperature = learn_temperature
            if learn_temperature == 'no':
                self.temperature = torch.tensor(initial_temperature, requires_grad=False, device=device)

            elif learn_temperature == 'unconditioned':
                self.temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature, device=device)), requires_grad=True)
                self.temperature_activation = torch.exp

            elif learn_temperature == 'conditioned':
                self.max_temperature = initial_temperature

                self.inverse_temperature_estimator = nn.Linear(output_dim, 1)
                self.inverse_temperature_activation = self.inverse_temperature_activation

        if self.sample_train == 'sparsemax' or self.sample_infer == 'sparsemax':
            self.sparsemax = Sparsemax()

        self.current_temperature = None

    def inverse_temperature_activation(self, inv_temp):
        inverse_max_temperature = 1. / self.max_temperature
        return torch.log(1 + torch.exp(inv_temp)) + inverse_max_temperature

    def update_temperature(self, decoder_states):
        if self.learn_temperature == 'no':
            self.current_temperature = self.temperature

        elif self.learn_temperature == 'unconditioned':
            self.current_temperature = self.temperature_activation(self.temperature)

        elif self.learn_temperature == 'conditioned':
            batch_size          = decoder_states.size(0)
            max_decoder_length  = decoder_states.size(1)
            hidden_dim          = decoder_states.size(2)

            estimator_input = decoder_states.view(batch_size * max_decoder_length, hidden_dim)
            inverse_temperature = self.inverse_temperature_activation(self.inverse_temperature_estimator(estimator_input))
            self.current_temperature = 1. / inverse_temperature

    def sample(self, attn, mask):
        batch_size, output_size, input_size = attn.size()

        # We are in training mode
        if self.training:
            if self.sample_train == 'full':
                attn = F.softmax(attn, dim=2)

            elif self.sample_train == 'full_hard':
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)

                mask = mask.expand(batch_size, output_size, input_size).contiguous().view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature, gumbel=False, eps=1e-20)
                attn = attn_hard.view(batch_size, -1, input_size) 

            elif 'gumbel' in self.sample_train:
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)
                mask = mask.expand(batch_size, output_size, input_size).contiguous().view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature, gumbel=True, eps=1e-20)
                
                if self.sample_train == 'gumbel_soft':
                    attn = attn_soft.view(batch_size, -1, input_size)
                elif self.sample_train == 'gumbel_hard':
                    attn = attn_hard.view(batch_size, -1, input_size) 

            elif self.sample_train == 'sparsemax':
                # Sparsemax only handles 2-dim tensors,
                # so we reshape and reshape back after sparsemax
                original_size = attn.size()
                attn = attn.view(-1, attn.size(2))
                attn = self.sparsemax(attn)
                attn = attn.view(original_size)

        # Inference mode
        else:
            if self.sample_infer == 'full':
                attn = F.softmax(attn, dim=2)

            elif self.sample_infer == 'full_hard':
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)
                mask = mask.expand(batch_size, output_size, input_size).contiguous().view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature, gumbel=False, eps=1e-20)
                attn = attn_hard.view(batch_size, -1, input_size) 
                
            elif 'gumbel' in self.sample_infer:
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)
                mask = mask.expand(batch_size, output_size, input_size).contiguous().view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature, gumbel=True, eps=1e-20)
                
                if self.sample_infer == 'gumbel_soft':
                    attn = attn_soft.view(batch_size, -1, input_size)
                elif self.sample_infer == 'gumbel_hard':
                    attn = attn_hard.view(batch_size, -1, input_size) 

            elif self.sample_infer == 'argmax':
                argmax = attn.argmax(dim=2, keepdim=True)
                attn = torch.zeros_like(attn)
                attn.scatter_(dim=2, index=argmax, value=1)

            elif self.sample_infer == 'sparsemax':
                # Sparsemax only handles 2-dim tensors,
                # so we reshape and reshape back after sparsemax
                original_size = attn.size()
                attn = attn.view(-1, attn.size(2))
                attn = self.sparsemax(attn)
                attn = attn.view(original_size)

        return attn

    # TODO: I actually think we should refactor Attention in Master to at least allow arguments
    # key, value, query. sample_method might be a bit specific though.
    def forward(self, attn, mask, queries):
        if (self.training and 'gumbel' in self.sample_train) or \
           (not self.training and 'gumbel' in self.sample_infer) or \
           (self.training and self.sample_train == 'full_hard') or \
           (not self.training and  self.sample_infer == 'full_hard'):
                self.update_temperature(queries)

        # TODO: Double, triple quadruple check whether the mask is correct, we don't take softmax more than once, etc.
        attn = self.sample(attn, mask)

        number_of_attention_vectors = attn.size(0) * attn.size(1)
        eps = 1e-2 * number_of_attention_vectors
        assert abs(torch.sum(attn) - number_of_attention_vectors) < eps, "Sum: {}, Number of attention vectors: {}".format(torch.sum(attn), number_of_attention_vectors)

        return attn