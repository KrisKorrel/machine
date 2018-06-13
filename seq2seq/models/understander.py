from __future__ import print_function, division

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..util.gumbel import gumbel_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Understander(nn.Module):

    """
    Seq2seq understander model with attention. Trained using reinforcement learning.
    First, pass the input sequence to `select_actions()` to perform forward pass and retrieve the actions
    Next, calculate and pass the rewards for the selected actions.
    Finally, call `finish_episod()` to calculate the discounted rewards and policy loss.
    """

    def __init__(self, rnn_cell, input_vocab_size, embedding_dim, hidden_dim, gamma, train_method, sample_train, sample_infer, initial_temperature, learn_temperature, attn_keys):
        """
        Args:
            input_vocab_size (int): Total size of the input vocabulary
            embedding_dim (int): Number of units to use for the input symbol embeddings
            hidden_dim (int): Size of the RNN cells in both encoder and decoder
            gamma (float): Gamma value to use for the discounted rewards
        """
        super(Understander, self).__init__()

        rnn_cell = rnn_cell.lower()

        self.encoder = UnderstanderEncoder(
            rnn_cell=rnn_cell,
            input_vocab_size=input_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim)

        # The attention scores are calculated from a concatenation of the decoder hidden state and the keys.
        # So we must pass the dimensions of the keys to the decoder
        if 'embeddings' in attn_keys:
            key_dim = embedding_dim
        elif 'outputs' in attn_keys:
            key_dim = hidden_dim

        self.decoder = UnderstanderDecoder(
            rnn_cell=rnn_cell,
            hidden_dim=hidden_dim,
            key_dim=key_dim)

        self.gamma = gamma

        self._saved_log_probs = []
        self._rewards = []

        self.train_method = train_method
        self.sample_train = sample_train
        self.sample_infer = sample_infer

        self.learn_temperature = learn_temperature
        if learn_temperature == 'no':
            self.temperature = torch.tensor(initial_temperature, requires_grad=False, device=device)

        elif learn_temperature == 'unconditioned':
            self.temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature, device=device)), requires_grad=True)
            self.temperature_activation = torch.exp

        elif learn_temperature == 'conditioned':
            max_temperature = initial_temperature

            inverse_max_temperature = 1. / max_temperature
            self.inverse_temperature_estimator = nn.Linear(hidden_dim,1)
            self.inverse_temperature_activation = lambda inv_temp: torch.log(1 + torch.exp(inv_temp)) + inverse_max_temperature
        self.current_temperature = None

        self.attn_keys = attn_keys

    def forward(self, state, valid_action_mask, max_decoding_length, possible_attn_keys):
        """
        Perform a forward pass through the seq2seq model

        Args:
            state (torch.tensor): [batch_size x max_input_length] tensor containing indices of the input sequence
            valid_action_mask (torch.tensor): [batch_size x max_input_length] ByteTensor containing a 1 for all non-pad inputs
            max_decoding_length (int): Maximum length till which the decoder should run

        Returns:
            torch.tensor: [batch_size x max_output_length x max_input_length] tensor containing the log-probabilities for each decoder step to attend to each encoder step
        """
        encoder_embeddings, encoded_hidden, encoder_outputs = self.encoder(input_variable=state)

        # Create dict with both understander and executor's encoder embeddings and outputs
        possible_attn_keys['understander_encoder_embeddings'] = encoder_embeddings
        possible_attn_keys['understander_encoder_outputs'] = encoder_outputs

        # Pick the correct attention keys
        attn_keys = possible_attn_keys[self.attn_keys] 

        action_logits, decoder_states = self.decoder(encoder_outputs=encoder_outputs, hidden=encoded_hidden, output_length=max_decoding_length, valid_action_mask=valid_action_mask, attn_keys=attn_keys)

        if  (self.training and 'gumbel' in self.sample_train) or \
            (not self.training and 'gumbel' in self.sample_infer):
            if self.learn_temperature == 'no':
                self.current_temperature = self.temperature

            elif self.learn_temperature == 'unconditioned':
                self.current_temperature = self.temperature_activation(self.temperature)

            elif self.learn_temperature == 'conditioned':
                # TODO: (max) decoder length?
                batch_size          = decoder_states.size(0)
                max_decoder_length  = decoder_states.size(1)
                hidden_dim          = decoder_states.size(2)

                estimator_input = decoder_states.view(batch_size * max_decoder_length, hidden_dim)
                inverse_temperature = self.inverse_temperature_activation(self.inverse_temperature_estimator(estimator_input))
                self.current_temperature = 1. / inverse_temperature

        return action_logits, possible_attn_keys

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

    def select_actions(self, state, input_lengths, max_decoding_length, epsilon, possible_attn_keys):
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
        if self._rewards:
            raise Exception("Did you forget to finish the episode?")

        # First, we establish which encoder states are valid to attend to.
        valid_action_mask = self.get_valid_action_mask(state, input_lengths)
        
        # We perform a forward pass to get the log-probability of attending to each
        # encoder for each decoder
        action_log_probs, possible_attn_keys = self.forward(state, valid_action_mask, max_decoding_length, possible_attn_keys)

        # In RL settings, we want to stochastically choose a single action.
        if self.train_method == 'rl':
            actions = []
            for decoder_step in range(max_decoding_length):
                # Get the log-probabilities for a single decoder time step
                # (batch_size x max_encoder_states)
                action_log_probs_current_step = action_log_probs[:, decoder_step, :]

                # In training mode:
                # Chance epsilon: Stochastically sample action from the policy
                # Chance 1-eps:   Stochastically sample action from uniform distribution
                if self.training:
                    categorical_distribution_policy = Categorical(logits=action_log_probs_current_step)

                    # Perform epsilon-greedy action sampling
                    sample = random.random()
                    # If we don't meet the epsilon threshold, we stochastically sample from the policy
                    if sample <= epsilon:
                        action = categorical_distribution_policy.sample()
                    # Else we sample the actions from a uniform distribution (over the valid actions)
                    else:
                        # We don't need to normalize these to probabilities, as this is already
                        # done in Categorical
                        uniform_probability_current_step = valid_action_mask.float()
                        categorical_distribution_uniform = Categorical(probs=uniform_probability_current_step)
                        action = categorical_distribution_uniform.sample()

                    log_prob = categorical_distribution_policy.log_prob(action)

                # In inference mode: Just use greedy policy (argmax)
                else:
                    log_prob, action = action_log_probs_current_step.max(dim=1)
                    action = action.long()

                # Append the action to the list of actions and store the log-probabilities of the chosen actions
                actions.append(action)
                self._saved_log_probs.append(log_prob)
        
            # Convert list into tensor and make it batch-first
            actions = torch.stack(actions).transpose(0, 1)

        # In supervised training, we need to have a differentiable sample (at train time)
        elif self.train_method == 'supervised':
            attn = action_log_probs

            batch_size          = attn.size(0)
            n_decoder_states    = attn.size(1)
            n_encoder_states    = attn.size(2)

            # We are in training mode
            if self.training:
                if self.sample_train == 'full':
                    attn = attn

                elif 'gumbel' in self.sample_train:
                    invalid_action_mask = valid_action_mask.eq(0).unsqueeze(1).expand(batch_size, n_decoder_states, n_encoder_states).contiguous().view(-1, n_encoder_states)
                    attn = attn.view(-1, n_encoder_states)
                    attn_hard, attn_soft = gumbel_softmax(logits=attn, invalid_action_mask=invalid_action_mask, hard=True, tau=self.current_temperature, eps=1e-20)
                    
                    if self.sample_train == 'gumbel_soft':
                        attn = attn_soft.view(batch_size, -1, n_encoder_states)
                    elif self.sample_train == 'gumbel_hard':
                        attn = attn_hard.view(batch_size, -1, n_encoder_states) 

            # Inference mode
            else:
                if self.sample_infer == 'full':
                    attn = attn

                elif 'gumbel' in self.sample_infer:
                    invalid_action_mask = valid_action_mask.eq(0).unsqueeze(1).expand(batch_size, n_decoder_states, n_encoder_states).contiguous().view(-1, n_encoder_states)
                    attn = attn.view(-1, n_encoder_states)
                    attn_hard, attn_soft = gumbel_softmax(logits=attn, invalid_action_mask=invalid_action_mask, hard=True, tau=self.current_temperature, eps=1e-20)
                    
                    if self.sample_infer == 'gumbel_soft':
                        attn = attn_soft.view(batch_size, -1, n_encoder_states)
                    elif self.sample_infer == 'gumbel_hard':
                        attn = attn_hard.view(batch_size, -1, n_encoder_states) 

                elif self.sample_infer == 'argmax':
                    argmax = attn.argmax(dim=2, keepdim=True)
                    attn = torch.zeros_like(attn)
                    attn.scatter_(dim=2, index=argmax, value=1)

            # In supervised setting we have as actions the entire attention vector(s)
            actions = attn

        return actions, possible_attn_keys

    def set_rewards(self, rewards):
        self._rewards = rewards

    def finish_episode(self):
        """
        Calculate discounted reward of entire episode and return policy loss


        Returns:
            torch.tensor: Single float representing the policy loss
        """

        # In inference mode, no rewards are added and we don't have to do anything besides reset.
        if not self._rewards:
            del self._rewards[:]
            del self._saved_log_probs[:]
            return None

        assert len(self._rewards) == len(self._saved_log_probs), "Number of rewards ({}) must equal number of actions ({})".format(len(self._rewards), len(self._saved_log_probs))

        # Calculate discounted rewards
        R = 0
        discounted_rewards = []

        # Get numpy array (n_rewards x batch_size)
        rewards = np.array(self._rewards)

        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R.tolist())

        discounted_rewards = torch.tensor(discounted_rewards, requires_grad=False, device=device)

        # TODO: This doesn't work when reward is negative
        # Normalize rewards
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=0, keepdim=True)) / \
        #     (discounted_rewards.std(dim=0, keepdim=True) + float(np.finfo(np.float32).eps))

        # (n_rewards x batch_size) -> (batch_size x n_rewards)
        discounted_rewards = discounted_rewards.transpose(0, 1)

        # Stack list of tensors to tensor of (batch_size x n_rewards)
        saved_log_probs = torch.stack(self._saved_log_probs, dim=1)

        # Calculate policy loss
        # Multiply each reward with it's negative log-probability element-wise
        policy_loss = -saved_log_probs * discounted_rewards
        
        # Sum over rewards, take mean over batch
        # TODO: Should we take mean over rewards?
        policy_loss = policy_loss.sum(dim=1).mean()

        # Reset episode
        del self._rewards[:]
        del self._saved_log_probs[:]

        return policy_loss


class UnderstanderEncoder(nn.Module):
    def __init__(self, rnn_cell, input_vocab_size, embedding_dim, hidden_dim):
        """
        Args:
            input_vocab_size (int): Total size of the input vocabulary
            embedding_dim (int): Number of units to use for the input symbol embeddings
            hidden_dim (int): Size of the RNN cells

        """
        super(UnderstanderEncoder, self).__init__()

        self.rnn_cell = rnn_cell
        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)

        # We will learn the initial hidden state
        if self.rnn_cell == 'lstm':
            h_0 = torch.zeros(self.n_layers, 1, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.n_layers, 1, self.hidden_dim, device=device)

            self.h_0 = nn.Parameter(h_0, requires_grad=True)
            self.c_0 = nn.Parameter(c_0, requires_grad=True)

            rnn_cell = nn.LSTM

        elif self.rnn_cell == 'gru':
            h_0 = torch.zeros(self.n_layers, 1, self.hidden_dim, device=device)

            self.h_0 = nn.Parameter(h_0, requires_grad=True)

            rnn_cell = nn.GRU

        self.encoder = rnn_cell(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=self.n_layers,
            batch_first=True)


    def forward(self, input_variable):
        """
        Forward propagation

        Args:
            input_variable (torch.tensor): [batch_size x max_input_length] tensor containing indices of the input sequence

        Returns:
            torch.tensor: The outputs of all encoder states
            torch.tensor: The hidden state of the last encoder state
        """
        input_embedding = self.input_embedding(input_variable)

        # Expand learned initial states to the batch size
        batch_size = input_embedding.size(0)
        if self.rnn_cell == 'lstm':
            h_0_batch = self.h_0.repeat(1, batch_size, 1)
            c_0_batch = self.c_0.repeat(1, batch_size, 1)
            hidden0 = (h_0_batch, c_0_batch)

        elif self.rnn_cell == 'gru':
            h_0_batch = self.h_0.repeat(1, batch_size, 1)
            hidden0 = h_0_batch

        out, hidden = self.encoder(input_embedding, hidden0)

        return input_embedding, hidden, out


class UnderstanderDecoder(nn.Module):

    """
    Decoder of the understander model. It will forward a concatenation of each combination of
    decoder state and encoder state through a MLP with 1 hidden layer to produce a score.
    We take the log-softmax of this to calculate the logits. All encoder states that are associated
    with <pad> inputs are not taken into account for calculations.
    """

    def __init__(self, rnn_cell, hidden_dim, key_dim):
        """      
        Args:
            hidden_dim (int): Size of the RNN cells
        """
        super(UnderstanderDecoder, self).__init__()

        self.embedding_dim = 1
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        # TODO: We don't have an embedding layer for now, as I'm not sure what the input to the
        # decoder should be. Maybe the last output? Maybe the hidden state of the executor?
        # For now I use a constant zero vector
        # self.input_embedding = nn.Embedding(1, embedding_dim)
        self.embedding = torch.zeros(1, self.n_layers, self.embedding_dim, requires_grad=False, device=device)

        self.rnn_cell = rnn_cell
        if self.rnn_cell == 'lstm':
            rnn_cell = nn.LSTM
        elif self.rnn_cell == 'gru':
            rnn_cell = nn.GRU

        self.decoder = rnn_cell(1, hidden_dim, batch_first=True)

        # Hidden layer of the MLP. Goes from dec_state_dim + key_dim to hidden dim
        self.hidden_layer = nn.Linear(hidden_dim + key_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()

        # Final layer that produces the log-probabilities
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_activation = nn.LogSoftmax(dim=2)

    def forward(self, encoder_outputs, hidden, output_length, valid_action_mask, attn_keys):
        """
        Forward propagation

        Args:
            encoder_outputs (torch.tensor): [batch_size x enc_len x enc_hidden_dim] output of all encoder states
            hidden (torch.tensor): ([batch_size x enc_hidden_dim] [batch_size x enc_hidden_dim]) h,c of last encoder state
            output_length (int): Predefined decoder length
            valid_action_mask (torch.tensor): ByteTensor with a 0 for each encoder state that is associated with <pad> input

        Returns:
            torch.tensor: [batch_size x dec_len x enc_len] Log-probabilities of choosing each encoder state for each decoder state
        """

        action_scores_list = []
        batch_size = encoder_outputs.size(0)

        # First decoder state should have as prev_hidden th hidden state of the encoder
        decoder_hidden = hidden

        decoder_states_list = []

        # TODO: We use rolled out version. If we actually won't use any (informative) input to the decoder
        # we should roll it to save computation time and have cleaner code.
        for decoder_step in range(output_length):
            # Expand the embedding to the batch
            embedding = self.embedding.expand(batch_size, self.n_layers, self.embedding_dim)

            # Forward propagate the decoder
            _, decoder_hidden = self.decoder(embedding, decoder_hidden)

            # We use the same MLP method as in attention.py
            encoder_states = attn_keys
            if self.rnn_cell == 'lstm':
                h, c = decoder_hidden # Unpack LSTM state
            elif self.rnn_cell == 'gru':
                h = decoder_hidden
            h = h.transpose(0, 1) # make it batch-first
            decoder_states = h

            # Store decoder state to return. Since we have unrolled decoder, we can squeeze the second dimension
            decoder_states_list.append(decoder_states.squeeze(1))

            # apply mlp to all encoder states for current decoder
            # decoder_states --> (batch, dec_seqlen, hl_size)
            # encoder_states --> (batch, enc_seqlen, hl_size)
            batch_size, enc_seqlen, hl_size_enc = encoder_states.size()
            _,          dec_seqlen, hl_size_dec       = decoder_states.size()

            # For the encoder states we add extra dimension with dec_seqlen
            # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
            encoder_states_exp = encoder_states.unsqueeze(1)
            encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size_enc)
            
            # For the decoder states we add extra dimension with enc_seqlen
            # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
            decoder_states_exp = decoder_states.unsqueeze(2)
            decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size_dec)
            
            # reshape encoder and decoder states to allow batchwise computation. We will have
            # in total batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
            # layer for each of them
            decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size_dec)
            encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size_enc)
            
            # tensor with two dimensions. The first dimension is the number of batchs which is:
            # batch_size x enc_seqlen x dec_seqlen
            # the second dimension is enc_hidden_dim + dec_hidden_dim
            mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)
            
            # apply mlp and reshape to get back in correct shape
            mlp_hidden = self.hidden_layer(mlp_input)
            mlp_hidden = self.hidden_activation(mlp_hidden)
            
            mlp_out = self.output_layer(mlp_hidden)
            mlp_out = mlp_out.view(batch_size, dec_seqlen, enc_seqlen)

            action_scores_list.append(mlp_out)

        # Combine the action scores for each decoder step into 1 variable
        action_scores = torch.cat(action_scores_list, dim=1)

        # Combine the decoder state for each decoder step into 1 variable
        decoder_states = torch.stack(decoder_states_list, dim=1)

        # Fill all invalid <pad> encoder states with 0 probability (-inf pre-softmax score)
        invalid_action_mask = valid_action_mask.ne(1).unsqueeze(1).expand(-1, output_length, -1)
        action_scores.masked_fill_(invalid_action_mask, -float('inf'))

        # For each decoder step, take the log-softmax over all actions to get log-probabilities
        action_logits = self.output_activation(action_scores)

        return action_logits, decoder_states
