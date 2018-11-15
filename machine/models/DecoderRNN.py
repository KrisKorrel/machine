import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, HardGuidance, ProvidedAttentionVectors
from .baseRNN import BaseRNN
from .seq2attn import Seq2attn
from .DecoderRNNModel import DecoderRNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
        full_focus(bool, optional): flag indication whether to use full attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id, embedding_dim,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, attention_method=None, full_focus=False,
                 sample_train=None,
                 sample_infer=None,
                 initial_temperature=None,
                 learn_temperature=None,
                 attn_keys=None,
                 attn_vals=None,
                 full_attention_focus='no'):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.rnn_type = rnn_cell
        self.max_length = max_len
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.n_layers = n_layers
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if attn_keys == 'embeddings':
            self.attn_keys = 'encoder_embeddings'
        elif attn_keys == 'outputs':
            self.attn_keys = 'encoder_outputs'

        if attn_vals == 'embeddings':
            self.attn_vals = 'encoder_embeddings'
        elif attn_vals == 'outputs':
            self.attn_vals = 'encoder_outputs'

        # increase input size decoder if attention is applied before decoder rnn
        if True:
            self.decoder_model = Seq2attn(
                rnn_cell=rnn_cell,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_size,
                output_dim=vocab_size,
                n_layers=n_layers,
                dropout_p=dropout_p,
                use_attention=use_attention,
                attention_method=attention_method,
                sample_train=sample_train,
                sample_infer=sample_infer,
                initial_temperature=initial_temperature,
                learn_temperature=learn_temperature,
                attn_keys=attn_keys,
                attn_vals=attn_vals,
                full_focus=full_focus,
                full_attention_focus=full_attention_focus)

        else:
            # TODO: Currently we do not use this anymore. We use seq2attn for baseline as well.
            self.decoder_model = DecoderRNNModel(vocab_size, max_len, hidden_size, sos_id, eos_id, n_layers,
                                             rnn_cell, bidirectional, input_dropout_p, dropout_p, use_attention, attention_method, full_focus)

            assert attn_keys == attn_vals == 'outputs', "For the baseline, only regular attention is supported"

        # If we initialize the executor's decoder with a new vector instead of the last encoder state
        # We initialize it as parameter here.
        if self.rnn_type == 'lstm':
            self.executor_hidden0 = (
                nn.Parameter(torch.zeros([self.n_layers, 1, self.hidden_size], device=device)),
                nn.Parameter(torch.zeros([self.n_layers, 1, self.hidden_size], device=device)))

        elif self.rnn_type == 'gru':
            self.executor_hidden0 = nn.Parameter(torch.zeros([self.n_layers, 1, self.hidden_size], device=device))

        if use_attention == 'post-rnn':
            self.out = nn.Linear(2 * self.hidden_size, vocab_size)
        else:
            self.out = nn.Linear(self.hidden_size, vocab_size)
            if full_focus:
                self.ffocus_merge = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward_step(self, input_var, seq2attn_decoder_hidden, executor_decoder_hidden, attn_keys, attn_vals, **attention_method_kwargs):
        """
        Performs one or multiple forward decoder steps.
        
        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs of the decoder RNN
        
        Returns:
            predicted_softmax: The output softmax distribution at every time step of the decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN
        """
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        # TODO: We should not have an if-else statement here. Should be agnostic of underlying recurrent model
        # To accomodate pondering, we just pass only 1 hidden state.
        # This will be disassembled again in the seq2attn
        if self.rnn_type == 'gru':
            ponder_hidden = torch.cat([seq2attn_decoder_hidden,
                                       executor_decoder_hidden], dim=2)
        elif self.rnn_type == 'lstm':
            ponder_hidden = (torch.cat([seq2attn_decoder_hidden[0],
                                        executor_decoder_hidden[0]], dim=2),
                             torch.cat([seq2attn_decoder_hidden[1],
                                        executor_decoder_hidden[1]], dim=2))
        return_values = self.decoder_model(
            embedded,
            ponder_hidden,
            attn_keys=attn_keys,
            attn_vals=attn_vals,
            **attention_method_kwargs)

        new_return_values = [F.log_softmax(self.out(return_values[0].contiguous().view(batch_size, -1)), dim=1).view(batch_size, output_size, -1)]
        for i in range(1, len(return_values)):
            new_return_values.append(return_values[i])

        return new_return_values

    def forward(self, inputs=None,
                encoder_embeddings=None, encoder_hidden=None, encoder_outputs=None,
                teacher_forcing_ratio=0, provided_attention=None, provided_attention_vectors=None, possible_attn_vals=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             teacher_forcing_ratio)        

        seq2attn_decoder_hidden = self._init_state(encoder_hidden, 'encoder')
        executor_decoder_hidden = self._init_state(encoder_hidden, 'new')

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                if not isinstance(step_attn, list):
                    step_attn = [step_attn]
                for s in step_attn:
                    ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(s)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Prepare extra arguments for attention method
        attention_method_kwargs = {}
        if self.decoder_model.attention and isinstance(self.decoder_model.attention.method, HardGuidance):
            attention_method_kwargs['provided_attention'] = provided_attention
        if self.decoder_model.attention and isinstance(self.decoder_model.attention.method, ProvidedAttentionVectors):
            attention_method_kwargs['provided_attention_vectors'] = provided_attention_vectors
            attention_method_kwargs['attn_vals'] = possible_attn_vals[self.attn_vals]

        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the attention based on
        # the previous hidden state, before we can calculate the next hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the decoder steps
        # one-by-one since the output needs to be copied to the input of the next step.
        # TODO: Currently we always unroll
        if self.use_attention == 'pre-rnn' or True or not use_teacher_forcing:
            unrolling = True
        else:
            unrolling = False

        # Get local variable out of locals() dictionary by string key
        attn_keys = locals()[self.attn_keys]
        attn_vals = locals()[self.attn_vals]

        if unrolling:
            symbols = None
            for di in range(max_length):
                # We always start with the SOS symbol as input. We need to add extra dimension of length 1 for the number of decoder steps (1 in this case)
                # When we use teacher forcing, we always use the target input.
                if di == 0 or use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)
                # If we don't use teacher forcing (and we are beyond the first SOS step), we use the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                if self.decoder_model.attention and (isinstance(self.decoder_model.attention.method, HardGuidance) or isinstance(self.decoder_model.attention.method, ProvidedAttentionVectors)):
                    attention_method_kwargs['step'] = di

                return_values = self.forward_step(decoder_input,
                                                  seq2attn_decoder_hidden,
                                                  executor_decoder_hidden,
                                                  attn_keys,
                                                  attn_vals,
                                                  **attention_method_kwargs)

                executor_decoder_output, ponder_decoder_hidden, step_attn = return_values[:3]

                # Decouple the ponder hidden state
                # IF we use seq2attn the seq2attn and executor are concatenated into 1 vector, we should slice this
                # TODO: If true..
                if True:
                    if self.rnn_type == 'gru':
                        seq2attn_decoder_hidden = ponder_decoder_hidden[:, :, :self.hidden_size].contiguous()
                        executor_decoder_hidden = ponder_decoder_hidden[:, :, self.hidden_size:].contiguous()
                    elif self.rnn_type == 'lstm':
                        seq2attn_decoder_hidden = (ponder_decoder_hidden[0][:, :, :self.hidden_size].contiguous(),
                                                       ponder_decoder_hidden[1][:, :, :self.hidden_size].contiguous())
                        executor_decoder_hidden = (ponder_decoder_hidden[0][:, :, self.hidden_size:].contiguous(),
                                                   ponder_decoder_hidden[1][:, :, self.hidden_size:].contiguous())
                # For the baseline model, there is / should be no seq2attn 
                else:
                    executor_decoder_hidden = ponder_decoder_hidden
                    seq2attn_decoder_hidden = None

                if not isinstance(step_attn, list):
                    step_attn = [(step_attn,)]

                step_attn = [s[0] for s in step_attn]

                # Remove the unnecessary dimension.
                step_output = executor_decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn)

                # print(torch.stack(step_attn).transpose(0, 1).squeeze(2)[0])
                # print("\n")
        else:
            # Remove last token of the longest output target in the batch. We don't have to run the last decoder step where the teacher forcing input is EOS (or the last output)
            # It still is run for shorter output targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            if self.decoder_model.attention and (isinstance(self.decoder_model.attention.method, HardGuidance) or isinstance(self.decoder_model.attention.method, ProvidedAttentionVectors)):
                attention_method_kwargs['step'] = -1

            executor_decoder_output, ponder_decoder_hidden, attn = self.forward_step(decoder_input,
                                              seq2attn_decoder_hidden,
                                              executor_decoder_hidden,
                                              attn_keys,
                                              attn_vals,
                                              **attention_method_kwargs)

            # Decouple the ponder hidden state
            # IF we use seq2attn the seq2attn and executor are concatenated into 1 vector, we should slice this
            # TODO: If true..
            if True:
                if self.rnn_type == 'gru':
                    seq2attn_decoder_hidden = ponder_decoder_hidden[:, :, :self.hidden_size].contiguous()
                    executor_decoder_hidden = ponder_decoder_hidden[:, :, self.hidden_size:].contiguous()
                elif self.rnn_type == 'lstm':
                    seq2attn_decoder_hidden = (ponder_decoder_hidden[0][:, :, :self.hidden_size].contiguous(),
                                                   ponder_decoder_hidden[1][:, :, :self.hidden_size].contiguous())
                    executor_decoder_hidden = (ponder_decoder_hidden[0][:, :, self.hidden_size:].contiguous(),
                                               ponder_decoder_hidden[1][:, :, self.hidden_size:].contiguous())
            # For the baseline model, there is / should be no seq2attn 
            else:
                executor_decoder_hidden = ponder_decoder_hidden
                seq2attn_decoder_hidden = None

            for di in range(executor_decoder_output.size(1)):
                step_output = executor_decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)

        # print(torch.stack(ret_dict[DecoderRNN.KEY_ATTN_SCORE]).squeeze()[0:4])
        # print("\n")

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, executor_decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden, init_dec_with):
        if init_dec_with == 'encoder':
            """ Initialize the encoder hidden state. """
            if encoder_hidden is None:
                return None
            if isinstance(encoder_hidden, tuple):
                encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
            else:
                encoder_hidden = self._cat_directions(encoder_hidden)

        elif init_dec_with == 'new':
            if isinstance(self.executor_hidden0, tuple):
                batch_size = encoder_hidden[0].size(1)
                encoder_hidden = (
                    self.executor_hidden0[0].repeat(1, batch_size, 1),
                    self.executor_hidden0[1].repeat(1, batch_size, 1))
            else:
                batch_size = encoder_hidden.size(1)
                encoder_hidden = self.executor_hidden0.repeat(1, batch_size, 1)

        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_type == 'lstm':
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_type == 'gru':
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.tensor([self.sos_id] * batch_size, dtype=torch.long, device=device).view(batch_size, 1)

            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
