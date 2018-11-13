"""
Computational model of single RNN decoder with optional attention methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
from .attention import Attention, HardGuidance


class DecoderRNNModel(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id, n_layers=1, rnn_cell='gru',
                 bidirectional=False, input_dropout_p=0, dropout_p=0, use_attention=False,
                 attention_method=None, full_focus=False):
        super(DecoderRNNModel, self).__init__(vocab_size, max_len,
                                              hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.input_size = hidden_size
        self.output_size = vocab_size

        self.use_attention = use_attention
        self.attention_method = attention_method
        self.full_focus = full_focus

        if use_attention != False and attention_method == None:
            raise ValueError("Method for computing attention should be provided")

        # increase input size decoder if attention is applied before decoder rnn
        if use_attention == 'pre-rnn' and not full_focus:
            self.input_size *= 2

        self.rnn = self.rnn_cell(self.input_size, self.hidden_size,
                                 self.n_layers, batch_first=True, dropout=self.dropout_p)

        if use_attention:
            self.attention = Attention(input_dim=self.hidden_size*2, output_dim=self.hidden_size, method=attention_method)
        else:
            self.attention = None

    def forward(self, embedded, hidden, encoder_outputs, **attention_method_kwargs):
        """
        Performs one or multiple forward decoder steps.

        Args:
            embedded (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs of the decoder RNN
            **attention_method_kwargs: Extra arguments for the attention method

        Returns:
            predicted_softmax: The output softmax distribution at every time step of the decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN
        """
        batch_size, output_size, _ = embedded.size()

        if self.use_attention == 'pre-rnn':
            h = hidden
            if isinstance(hidden, tuple):
                h, c = hidden
            # Apply the attention method to get the attention vector and weighted
            # context vector. Provide decoder step for hard attention
            # transpose to get batch at the second index
            context, attn = self.attention(
                queries=h[-1:].transpose(0, 1),
                keys=encoder_outputs,
                values=encoder_outputs,
                **attention_method_kwargs)
            combined_input = torch.cat((context, embedded), dim=2)
            if self.full_focus:
                merged_input = F.relu(self.ffocus_merge(combined_input))
                combined_input = torch.mul(context, merged_input)
            output, hidden = self.rnn(combined_input, hidden)

        elif self.use_attention == 'post-rnn':
            output, hidden = self.rnn(embedded, hidden)
            # Apply the attention method to get the attention vector and weighted
            # context vector. Provide decoder step for hard attention
            context, attn = self.attention(
                queries=output,
                keys=encoder_outputs,
                values=encoder_outputs,
                **attention_method_kwargs)
            output = torch.cat((context, output), dim=2)

        elif not self.use_attention:
            attn = None
            output, hidden = self.rnn(embedded, hidden)

        return output, hidden, attn
