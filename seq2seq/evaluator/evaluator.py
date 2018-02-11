from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss, Variance

import numpy as np

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data, writer=None, run_step=0):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()

        word_match = 0
        word_total = 0

        seq_match = 0
        seq_total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]
        #input_words = {k: [0.0] for k in range(1,8)}
        #att_dict = {j: dict(input_words) for j in range(1, 14)}
        att_dict = {}
        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            attentions = [att.squeeze() for att in other['attention_score']]

            for q, input_seq in enumerate(input_variables.data):
                prediction = [step.squeeze().data[q] for step in seqlist]
                for i, output_word in enumerate(target_variables[q][1:].data.cpu().numpy()):
                    if target_variables.data[q][i+1] != pad:
                        if output_word in att_dict:
                            output_word_dict = att_dict[output_word]
                        else:
                            output_word_dict = {}
                            att_dict[output_word] = output_word_dict
                        for j, input_word in enumerate(input_seq):
                            if input_word in output_word_dict:
                                output_word_dict[input_word].append(attentions[i][q][j].data.cpu().numpy()[0])
                            else:
                                output_word_dict[input_word] = [attentions[i][q][j].data.cpu().numpy()[0]]

            match_per_seq = torch.zeros(batch.batch_size).type(torch.FloatTensor)
            total_per_seq = torch.zeros(batch.batch_size).type(torch.FloatTensor)

            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)

                correct_per_seq = (seqlist[step].view(-1).eq(target).data + non_padding.data).eq(2)
                match_per_seq += correct_per_seq.type(torch.FloatTensor)
                total_per_seq += non_padding.type(torch.FloatTensor).data

            word_match += match_per_seq.sum()
            word_total += total_per_seq.sum()

            seq_match += match_per_seq.eq(total_per_seq).sum()
            seq_total += total_per_seq.shape[0]

        cooccurrences = np.array(
            [[np.mean(att_dict[output_word][input_word]) for input_word in sorted(att_dict[output_word])] for
             output_word in sorted(att_dict)])

        if writer is not None:
            cooccurrences_tensor = torch.Tensor(cooccurrences)
            writer.add_image("co-occurences", cooccurrences_tensor, run_step)

        variance = sum(np.var(cooccurrences, axis=0))

        if word_total == 0:
            accuracy = float('nan')
        else:
            accuracy = word_match / word_total

        if seq_total == 0:
            seq_accuracy = float('nan')
        else:
            seq_accuracy = seq_match/seq_total

        print("TODO: Evaluator loss and variance are not correct")

        return loss.get_loss() + 1000*(3.25 - variance), accuracy, seq_accuracy, variance
