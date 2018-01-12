from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size
        self.fig_index = 0

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        import torch.autograd as autograd
        import matplotlib.pyplot as plt

        for src_tok in data.fields[seq2seq.src_field_name].vocab.freqs:
            idx = [data.fields[seq2seq.src_field_name].vocab.stoi[src_tok]]
            idx_t = autograd.Variable(torch.LongTensor(idx))
            embed = model.encoder.ret_embed(idx_t)

            if src_tok in ['jump', 'run', 'walk', 'look']:
                c = 'green'
            elif src_tok in ['left', 'right']:
                c = 'blue'
            elif src_tok in ['after']:
                c = 'red'
            elif src_tok in ['and']:
                c = 'orange'
            elif src_tok in ['twice', 'thrice']:
                c = 'black'
            elif src_tok in ['around']:
                c = 'grey'
            elif src_tok in ['turn']:
                c = 'purple'
            elif src_tok in ['opposite']:
                c = 'brown'
            else:
                print(src_tok)

            plt.scatter(embed.data.cpu().numpy()[0][0], embed.data.cpu().numpy()[0][1], marker='x', color=c, s=20)
            plt.text(embed.data.cpu().numpy()[0][0], embed.data.cpu().numpy()[0][1], src_tok, color=c)
        plt.savefig('../pics/test{}'.format(self.fig_index))
        plt.clf()
        self.fig_index += 1


        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
