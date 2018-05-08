from __future__ import print_function
import math
import torch
import torch.nn as nn
import numpy as np

class Metric(object):
    """ Base class for encapsulation of the metrics functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Attributes:
        name (str): name of the metric used by logging messages.
        target (str): dictionary key to fetch the target from dictionary that stores
                      different variables computed during the forward pass of the model
        metric_total (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the value of the metric of multiple batches.
            sub-classes.
    """

    def __init__(self, name, log_name, input_var):
        self.name = name
        self.log_name = log_name
        self.input = input_var

    def reset(self):
        """ Reset accumulated metric values"""
        raise NotImplementedError("Implement in subclass")

    def get_val(self):
        """ Get the value for the metric given the accumulated loss
        and the normalisation term

        Returns:
            loss (float): value of the metric.
        """
        raise NotImplementedError("Implement in subclass")

    def eval_batch(self, outputs, target):
        """ Compute the metric for the batch given results and target results.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError("Implement in subclass")

class WordAccuracy(Metric):
    """
    Batch average of word accuracy.

    Args:
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Word Accuracy"
    _SHORTNAME = "acc"
    _INPUT = "sequence"

    def __init__(self, weight=None, mask=None):
        self.mask = mask
        self.weight = weight
        self.word_match = 0
        self.word_total = 0

        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(WordAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.word_total != 0:
            return float(self.word_match)/self.word_total
        else:
            return 0

    def reset(self):
        self.word_match = 0
        self.word_total = 0

    def eval_batch(self, outputs, targets):
        # evaluate batch
        batch_size = targets.size(0)

        for step, step_output in enumerate(outputs):
            target = targets[:, step + 1]
            non_padding = target.ne(self.mask)
            correct = outputs[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
            self.word_match += correct
            self.word_total += non_padding.sum().data[0]

class SequenceAccuracy(Metric):
    """
    Batch average of word accuracy.

    Args:
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Sequence Accuracy"
    _SHORTNAME = "seq_acc"
    _INPUT = "seqlist"

    def __init__(self, weight=None, mask=None):
        self.mask = mask
        self.weight = weight
        self.seq_match = 0
        self.seq_total = 0

        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(SequenceAccuracy, self).__init__(self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.seq_total != 0:
            return float(self.seq_match)/self.seq_total
        else:
            return 0

    def reset(self):
        self.seq_match = 0
        self.seq_total = 0

    def eval_batch(self, outputs, targets):

        batch_size = targets.size(0)

        # compute sequence accuracy over batch
        match_per_seq = torch.zeros(batch_size).type(torch.FloatTensor)
        total_per_seq = torch.zeros(batch_size).type(torch.FloatTensor)

        for step, step_output in enumerate(outputs):
            target = targets[:, step + 1]

            non_padding = target.ne(self.mask)

            correct_per_seq = (outputs[step].view(-1).eq(target).data + non_padding.data).eq(2)
            match_per_seq += correct_per_seq.type(torch.FloatTensor)
            total_per_seq += non_padding.type(torch.FloatTensor).data

        self.seq_match += match_per_seq.eq(total_per_seq).sum()
        self.seq_total += total_per_seq.shape[0]
