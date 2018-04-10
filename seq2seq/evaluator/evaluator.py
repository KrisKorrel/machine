from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss, AttentionLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
        ignore_output_eos (bool, optional): Whether to ignore the output EOS for loss and metrics calculation, (default: False)
    """

    def __init__(self, batch_size=64, losses=[NLLLoss()], metrics=[WordAccuracy(), SequenceAccuracy()], ignore_output_eos=False):
        self.batch_size = batch_size

        self.losses = losses
        self.attention_loss_used = any(isinstance(loss, AttentionLoss) for loss in self.losses)

        self.metrics = metrics

        self.ignore_output_eos = ignore_output_eos
        self.output_eos_token = None
        self.output_pad_token = None
        self.attention_function = None

    def update_batch_metrics(self, metrics, other, target_variable):
        """
        Update a list with metrics for current batch.

        Args:
            metrics (list): list with of seq2seq.metric.Metric objects
            other (dict): dict generated by forward pass of model to be evaluated
            target_variable (dict): map of keys to different targets of model

        Returns:
            metrics (list): list with updated metrics
        """
        # evaluate output symbols
        outputs = other['sequence']

        for metric in metrics:
            metric.eval_batch(outputs, target_variable)

        return metrics

    def compute_batch_loss(self, decoder_outputs, decoder_hidden, other, target_variable):
        """
        Compute the loss for the current batch.

        Args:
            decoder_outputs (torch.Tensor): decoder outputs of a batch
            decoder_hidden (torch.Tensor): decoder hidden states for a batch
            other (dict): maps extra outputs to torch.Tensors
            target_variable (dict): map of keys to different targets

        Returns:
           losses (list): a list with seq2seq.loss.Loss objects
        """

        losses = self.losses
        for loss in losses:
            loss.reset()

        losses = self.update_loss(losses, decoder_outputs, decoder_hidden, other, target_variable)

        return losses

    def update_loss(self, losses, decoder_outputs, decoder_hidden, other, target_variable):
        """
        Update a list with losses for current batch

        Args:
            losses (list): a list with seq2seq.loss.Loss objects
            decoder_outputs (torch.Tensor): decoder outputs of a batch
            decoder_hidden (torch.Tensor): decoder hidden states for a batch
            other (dict): maps extra outputs to torch.Tensors
            target_variable (dict): map of keys to different targets

        Returns:
           losses (list): a list with seq2seq.loss.Loss objects
        """

        for loss in losses:
            loss.eval_batch(decoder_outputs, other, target_variable)

        return losses

    def evaluate(self, model, data, ponderer=None, attention_function=None):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
            ponderer (seq2seq.trainer.PonderGenerator): Object that masks silent pondering steps, (defeault: None)
            attention_function (seq2seq.trainer.AttentionGenerator): Generator of ground-truth attention guidance, (default: None)
            
        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        # Store the eos and pad tokens of the output data
        self.output_eos_token = data.fields[seq2seq.tgt_field_name].vocab.stoi['<eos>']
        self.output_pad_token = data.fields[seq2seq.tgt_field_name].vocab.stoi['<pad>']

        self.attention_function = attention_function
        if self.attention_function is None:
            assert self.attention_loss_used is False, "Evaluator is supposed to calculate attention loss, but no attention function is provided"

        model.eval()

        losses = self.losses
        for loss in losses:
            loss.reset()

        metrics = self.metrics
        for metric in metrics:
            metric.reset()

        # create batch iterator
        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        # loop over batches
        for batch in batch_iterator:

            input_variable, input_lengths, target_variable = self.get_batch_data(batch)

            decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths.tolist(), target_variable['decoder_output'])

            # apply metrics over entire sequence
            metrics = self.update_batch_metrics(metrics, other, target_variable)

            # mask out silent steps in case of pondering
            if ponderer is not None:
                decoder_outputs = ponderer.mask_silent_outputs(input_variable, input_lengths, decoder_outputs)
                decoder_targets = ponderer.mask_silent_targets(input_variable, input_lengths, target_variable['decoder_output'])
                target_variable['decoder_output'] = decoder_targets

            losses = self.update_loss(losses, decoder_outputs, decoder_hidden, other, target_variable)

        accuracy = metrics[0].get_val()
        seq_accuracy = metrics[1].get_val()

        return losses, metrics

    def get_batch_data(self, batch):
        input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
        target_variables = {'decoder_output': getattr(batch, seq2seq.tgt_field_name)}

        # Replace all <eos> with <pad> in the output targets. This should make
        # sure that they are ignored in calculating the output loss
        if self.ignore_output_eos:
            eos_indices = (target_variables['decoder_output']==self.output_eos_token)
            target_variables['decoder_output'] = target_variables['decoder_output'].masked_fill(eos_indices, self.output_pad_token)

        # Add attention targets if attention guidance function is provided
        if self.attention_function is not None:
            target_variables = self.attention_function.add_attention_targets(input_variables, input_lengths, target_variables)

        return input_variables, input_lengths, target_variables