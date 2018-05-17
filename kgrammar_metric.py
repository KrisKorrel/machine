from __future__ import print_function

from seq2seq.metrics import Metric

import torch

class KGrammarAccuracy(Metric):
    """
    Batch average of k-grammar sequence accuracy.

    Args:
        ignore_index (int, optional): index of masked token
    """

    _NAME = "K-Grammar Accuracy"
    _SHORTNAME = "k_grammar_acc"
    _INPUT = "seqlist" #TODO: What?

    def __init__(self, input_vocab, output_vocab, input_pad_symbol, use_output_eos, output_sos_symbol, output_pad_symbol, output_eos_symbol, output_unk_symbol):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_pad_symbol = input_pad_symbol
        self.use_output_eos = use_output_eos
        self.output_sos_symbol = output_sos_symbol
        self.output_pad_symbol = output_pad_symbol
        self.output_eos_symbol = output_eos_symbol
        self.output_unk_symbol = output_unk_symbol

        self.seq_correct = 0
        self.seq_total = 0

        super(KGrammarAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.seq_total != 0:
            return float(self.seq_correct) / self.seq_total
        else:
            return 0

    def reset(self):
        self.seq_correct = 0
        self.seq_total = 0

    # Original code provided by the authors of The Fine Line between Linguistic Generalization and Failure in Seq2Seq-Attention Models (https://arxiv.org/pdf/1805.01445.pdf)
    def correct(grammar, prediction):
        '''
        Return True if the target is a valid output given the source
        Args
            src (str)  
            target (str)
        formats should match those in the datafiles
        '''

        grammar_vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AS', 'BS', 'CS', 'DS', 'ES', 'FS', 'GS', 'HS', 'IS', 'JS', 'KS', 'LS', 'MS', 'NS', 'OS']

        all_correct = False
        #Check if the length is correct
        length_check = True if len(prediction) == 3 * len(grammar) else False
        #Check if everything falls in the same bucket, and there are no repeats
        for idx, inp in enumerate(grammar):
            vocab_idx = grammar_vocab.index(inp) + 1
            span = prediction[idx*3:idx*3+3]

            span_str = " ".join(span)
            if (not all(int(item.replace("A", "").replace("B", "").replace("C", "").split("_")[0]) == vocab_idx for item in span)
                    or (not ("A" in span_str and "B" in span_str and "C" in span_str))):
                all_correct = False
                break
            else:
                all_correct = True
        return all_correct


    def eval_batch(self, outputs, targets):
        # batch_size X N variable containing the indices of the model's input,
        # where N is the longest input
        input_variable = targets['encoder_input']

        batch_size = input_variable.size(0)

        # Convert to batch_size x M variable containing the indices of the model's output, where M
        # is the longest output
        predictions = torch.stack(outputs, dim=1).view(batch_size, -1)

        # Current implementation does not allow batch-wise evaluation
        for i_batch_element in range(batch_size):
            # We start by counting the sequence to the total.
            # Next we go through multiple checks for incorrectness.
            # If all these test fail, we consider the sequence correct.
            self.seq_total += 1

            # Extract the current example
            grammar = input_variable[i_batch_element, :].data.cpu().numpy()
            prediction = predictions[i_batch_element, :].data.cpu().numpy()
            
            # Convert indices to strings
            # Remove all padding from the grammar.
            grammar = [self.input_vocab.itos[token] for token in grammar if token != self.input_vocab.itos[token] != self.input_pad_symbol]
            prediction = [self.output_vocab.itos[token] for token in prediction]

            # Input and output EOS are present
            if self.use_output_eos:
                # -1 for EOS
                required_output_length = 3 * (len(grammar) - 1)

                # prediction_length should be 3*grammar_length + 1 for the EOS
                # TODO: Is this even possible? Won't the decoder always decode to at least the same
                # length as the target_output?
                if len(prediction) < (required_output_length + 1):
                    print("Too short")
                    exit(0)
                    continue

                # Last prediction should be EOS
                if prediction[required_output_length] != self.output_eos_symbol:
                    continue

                # Remove EOS (and possible padding)
                grammar_correct_length = grammar[:-1]
                prediction_correct_length = prediction[:required_output_length]

                # Check whether the predicted output is too short
                if self.output_eos_symbol in prediction_correct_length:
                    continue

            # No input and output EOS should be present in the data
            else:
                required_output_length = 3 * len(grammar)

                # prediction_length should be 3*grammar_length
                # TODO: Is this even possible? Won't the decoder always decode to at least the same
                # length as the target_output?
                if len(prediction) < required_output_length:
                    print("Too short")
                    exit(0)
                    continue

                # Remove possible padding
                grammar_correct_length = grammar
                prediction_correct_length = prediction[:required_output_length]

            # Since the SOS and PAD token are members of the target vocabulary, the decoder can also predict
            # those at the beginning of training.
            if self.output_sos_symbol in prediction_correct_length or self.output_pad_symbol in prediction_correct_length or self.output_unk_symbol in prediction_correct_length:
                continue

            # Check whether the prediction comes from the grammar
            if correct(grammar_correct_length, prediction_correct_length):
                self.seq_correct += 1