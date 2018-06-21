import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_dec_dropout = nn.Dropout(p=0.5)
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward_encoder(self, input_variable, input_lengths=None):
        executor_encoder_embeddings, encoder_hidden, executor_encoder_outputs = self.encoder(input_variable, input_lengths)
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = self.enc_dec_dropout(encoder_hidden[0]), self.enc_dec_dropout(encoder_hidden[1])
        else:
            encoder_hidden = self.enc_dec_dropout(encoder_hidden)
        return executor_encoder_embeddings, encoder_hidden, executor_encoder_outputs

    def forward_decoder(self, target_variables, teacher_forcing_ratio, executor_encoder_embeddings, encoder_hidden, executor_encoder_outputs):
        # Unpack target variables
        target_output = target_variables.get('decoder_output', None)
        # The attention target is preprended with an extra SOS step. We must remove this
        provided_attention = target_variables['attention_target'][:, 1:] if 'attention_target' in target_variables else None
        provided_attention_vectors = target_variables.get('provided_attention_vectors', None)
        understander_encoder_embeddings = target_variables.get('understander_encoder_embeddings', None)
        understander_encoder_outputs = target_variables.get('understander_encoder_outputs', None)

        possible_attn_vals = {
            'understander_encoder_embeddings': understander_encoder_embeddings,
            'understander_encoder_outputs': understander_encoder_outputs,
            'executor_encoder_embeddings': executor_encoder_embeddings,
            'executor_encoder_outputs': executor_encoder_outputs
        }

        result = self.decoder(inputs=target_output,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=executor_encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              provided_attention=provided_attention,
                              provided_attention_vectors=provided_attention_vectors,
                              possible_attn_vals=possible_attn_vals)

        return result

    def forward(self, input_variable, input_lengths=None, target_variables=None,
                teacher_forcing_ratio=0):

        executor_encoder_embeddings, encoder_hidden, executor_encoder_outputs = self.forward_encoder(input_variable, input_lengths)
        result = self.forward_decoder(target_variables, teacher_forcing_ratio, executor_encoder_embeddings, encoder_hidden, executor_encoder_outputs)

        return result
