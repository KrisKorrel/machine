from __future__ import print_function, division

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Understander(nn.Module):

    """
    Seq2seq understander model with attention.
    First, pass the input sequence to `select_actions()` to perform forward pass and retrieve the actions
    Next, calculate and pass the rewards for the selected actions.
    Finally, call `finish_episod()` to calculate the discounted rewards and policy loss.
    """

    # TODO: Do we need attn_keys and vals here? Can't they just only be passed as variables in forward()?
    def __init__(self, model_type, rnn_cell, embedding_dim, n_layers, hidden_dim, output_dim, dropout_p, use_attention, attention_method, sample_train, sample_infer, initial_temperature, learn_temperature, attn_keys, attn_vals, full_focus, full_attention_focus):
        """
        Args:
            input_vocab_size (int): Total size of the input vocabulary
            embedding_dim (int): Number of units to use for the input symbol embeddings
            hidden_dim (int): Size of the RNN cells in both encoder and decoder
        """
        super(Understander, self).__init__()

        self.model_type = model_type
        self.hidden_size = hidden_dim
        self.full_attention_focus = (full_attention_focus == 'yes')

        # Get type of RNN cell
        rnn_cell = rnn_cell.lower()
        self.rnn_type = rnn_cell
        if rnn_cell == 'lstm':
            rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            rnn_cell = nn.GRU

        self.use_attention = use_attention
        if not self.use_attention:
            raise Exception("Current implementation requires attention")
        if self.model_type == 'seq2attn' and self.use_attention != 'pre-rnn':
            raise Exception("Must use pre-rnn in combination with seq2attn")
        if self.use_attention == 'post-rnn' and full_focus:
            raise Exception("Full focus can only be used with pre-rnn")

        # Input size is hidden_size + context vector size, which depends on the type of attention value
        # TODO: As Yann pointed out, we should have different embedding size for decoder.
        if 'embeddings' in attn_keys:
            key_dim = embedding_dim
        elif 'outputs' in attn_keys:
            key_dim = hidden_dim
        if 'embeddings' in attn_vals:
            val_dim = embedding_dim
        elif 'outputs' in attn_vals:
            val_dim = hidden_dim

        input_size = hidden_dim + val_dim

        # Initialize models
        if self.model_type == 'baseline' and self.use_attention == 'pre-rnn' and not full_focus:
            understander_input_size = 2 * hidden_dim
        else:
            understander_input_size = hidden_dim

        self.understander_decoder = rnn_cell(understander_input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout_p)
        self.attention = Attention(input_dim=hidden_dim+key_dim, output_dim=hidden_dim, method=attention_method, sample_train=sample_train, sample_infer=sample_infer, learn_temperature=learn_temperature, initial_temperature=initial_temperature)
        self.executor_decoder = rnn_cell(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout_p)

        self.full_focus = full_focus
        if self.full_focus:
            self.ffocus_merge = nn.Linear(input_size, self.hidden_size)

    def get_valid_action_mask(self, state, input_lengths):
        """
        Get a bytetensor that indicates which encoder states are valid to attend to.
        All <pad> steps are invalid
        
        Args:
            state (torch.tensor): [batch_size x max_input_length] input variable
            input_lengths (torch.tensor): [batch_size] tensor containing the input length of each sequence in the batch
        
        Returns:
            torch.tensor: [batch_size x max_input_length] ByteTensor with a 0 for all <pad> elements
        """
        batch_size = state.size(0)

        # First, we establish which encoder states are valid to attend to. For
        # this we use the input_lengths
        max_encoding_length = torch.max(input_lengths)

        # (batch_size) -> (batch_size x max_encoding_length)
        input_lengths_expanded = input_lengths.unsqueeze(1).expand(-1, max_encoding_length)

        # Use arange to create list 0, 1, 2, 3, .. for each element in the batch
        # (batch_size x max_encoding_length)
        encoding_steps_indices = torch.arange(max_encoding_length, dtype=torch.long, device=device)
        encoding_steps_indices = encoding_steps_indices.unsqueeze(0).expand(batch_size, -1)

        # A (batch_size x max_encoding_length) tensor that has a 1 for all valid
        # actions and 0 for all invalid actions
        valid_action_mask = encoding_steps_indices < input_lengths_expanded

        return valid_action_mask

    def get_context(self, queries, keys, values, **attention_method_kwargs):
        context, attn = self.attention(queries=queries, keys=keys, values=values, **attention_method_kwargs)
 
        batch_size, dec_seqlen, enc_seqlen = attn.size()

        return context, attn

    def forward(self, embedded, ponder_decoder_hidden, attn_keys, attn_vals, **attention_method_kwargs):
        """
        Perform forward pass and stochastically select actions using epsilon-greedy RL

        Args:
            state (torch.tensor): [batch_size x max_input_length] tensor containing indices of the input sequence
            input_lengths (list): List containing the input length for each element in the batch
            max_decoding_length (int): Maximum length till which the decoder should run
            epsilon (float): epsilon for epsilon-greedy RL. Set to 1 in inference mode

        Returns:
            list(torch.tensor): List of length max_output_length containing the selected actions
        """

        if self.rnn_type == 'gru':
            understander_decoder_hidden = ponder_decoder_hidden[:, :, :self.hidden_size].contiguous()
            executor_decoder_hidden = ponder_decoder_hidden[:, :, self.hidden_size:].contiguous()
        elif self.rnn_type == 'lstm':
            understander_decoder_hidden = (ponder_decoder_hidden[0][:, :, :self.hidden_size].contiguous(),
                                           ponder_decoder_hidden[1][:, :, :self.hidden_size].contiguous())
            executor_decoder_hidden = (ponder_decoder_hidden[0][:, :, self.hidden_size:].contiguous(),
                                       ponder_decoder_hidden[1][:, :, self.hidden_size:].contiguous())

        # First, we establish which encoder states are valid to attend to.
        # TODO: Only works when keys are (full) hidden states. Maybe we should pass the mask (set_mask)

        # We perform a forward pass to get the log-probability of attending to each
        # encoder for each decoder



        if self.model_type == 'seq2attn':
            understander_decoder_output, understander_decoder_hidden = self.understander_decoder(embedded, understander_decoder_hidden)
            context, attn = self.get_context(queries=understander_decoder_output, keys=attn_keys, values=attn_vals, **attention_method_kwargs)
            executor_decoder_input = torch.cat((context, embedded), dim=2)
            if self.full_focus:
                executor_decoder_input = F.relu(self.ffocus_merge(executor_decoder_input))
                executor_decoder_input = torch.mul(context, executor_decoder_input)
            if self.full_attention_focus:
                if self.rnn_type == 'gru':
                    executor_decoder_hidden = executor_decoder_hidden * context.transpose(0, 1)
                elif self.rnn_type == 'lstm':
                    executor_decoder_hidden = (executor_decoder_hidden[0] * context.transpose(0, 1),
                                               executor_decoder_hidden[1] * context.transpose(0, 1))
            executor_decoder_output, executor_decoder_hidden = self.executor_decoder(executor_decoder_input, executor_decoder_hidden)

            output = executor_decoder_output

        elif self.model_type == 'baseline':
            if self.use_attention == 'pre-rnn':
                # When using multilayer rnns we simply take the last layer of
                # the encoder as attention queries
                if self.rnn_type == 'gru':
                    if understander_decoder_hidden.size(0) > 1:
                        attn_queries = understander_decoder_hidden[-2:-1].transpose(0, 1)
                    else:
                        attn_queries = understander_decoder_hidden.transpose(0, 1)

                elif self.rnn_type == 'lstm':
                    if understander_decoder_hidden[0].size(0) > 1:
                        attn_queries = understander_decoder_hidden[0][-2:-1].transpose(0, 1)
                    else:
                        attn_queries = understander_decoder_hidden[0].transpose(0, 1)

                context, attn = self.get_context(queries=attn_queries, keys=attn_keys, values=attn_vals, **attention_method_kwargs)
                understander_decoder_input = torch.cat((context, embedded), dim=2)

                if self.full_focus:
                    understander_decoder_input = F.relu(self.ffocus_merge(understander_decoder_input))
                    understander_decoder_input = torch.mul(context, understander_decoder_input)
                if self.full_attention_focus:
                    understander_decoder_hidden = understander_decoder_hidden * context.transpose(0, 1)

                understander_decoder_output, understander_decoder_hidden = self.understander_decoder(understander_decoder_input, understander_decoder_hidden)

                output = understander_decoder_output

            elif self.use_attention == 'post-rnn':
                understander_decoder_output, understander_decoder_hidden = self.understander_decoder(embedded, understander_decoder_hidden)

                # When using multilayer rnns we simply take the last layer of
                # the encoder as attention queries
                if self.rnn_type == 'gru':
                    if understander_decoder_hidden.size(0) > 1:
                        attn_queries = understander_decoder_hidden[-2:-1].transpose(0, 1)
                    else:
                        attn_queries = understander_decoder_hidden.transpose(0, 1)

                elif self.rnn_type == 'lstm':
                    if understander_decoder_hidden[0].size(0) > 1:
                        attn_queries = understander_decoder_hidden[0][-2:-1].transpose(0, 1)
                    else:
                        attn_queries = understander_decoder_hidden[0].transpose(0, 1)

                context, attn = self.get_context(queries=attn_queries, keys=attn_keys, values=attn_vals, **attention_method_kwargs)
                executor_decoder_input = torch.cat((context, understander_decoder_output), dim=2)

                if self.full_attention_focus:
                    understander_decoder_hidden = understander_decoder_hidden * context.transpose(0, 1)

                output = executor_decoder_input

        # TODO: For pondering in combination with baseline: It currently also uses the executor, which it should not.
        if self.rnn_type == 'gru':
            ponder_hidden = torch.cat([understander_decoder_hidden,
                                       executor_decoder_hidden], dim=2)
        elif self.rnn_type == 'lstm':
            ponder_hidden = (torch.cat([understander_decoder_hidden[0],
                                        executor_decoder_hidden[0]], dim=2),
                             torch.cat([understander_decoder_hidden[1],
                                        executor_decoder_hidden[1]], dim=2))

        return output, ponder_hidden, attn
