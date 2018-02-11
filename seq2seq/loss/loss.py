from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np

class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss functions.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()

class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """

    _NAME = "Avg NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(NLLLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data[0]
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

class Perplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    _NAME = "Perplexity"
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None):
        super(Perplexity, self).__init__(weight=weight, mask=mask, size_average=False)

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.size())
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term
        if nll > Perplexity._MAX_EXP:
            print("WARNING: Loss exceeded maximum value, capping to e^100")
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)

class Variance(object):
    @staticmethod
    def get_variance(inputs, outputs, attentions, input_vocab_size, output_vocab_size, reg_scale):
        # create empty confusion matrix
        confusion_matrix = torch.zeros(output_vocab_size, input_vocab_size)
        if torch.cuda.is_available():
            pass
            # confusion_matrix = confusion_matrix.cuda()
        confusion_matrix = Variable(confusion_matrix)

        # loop over attention vectors
        output_step = 0
        for attention in attentions:
            
            # flatten and squeeze vector
            if torch.cuda.is_available():
                attention = attention.cpu()
            attention_flat = attention.contiguous().view(-1)

            attention = attention.squeeze()

            # compute confusion matrix indices for each attention value
            attention_size_total = attention_flat.size(0)
            indices = torch.LongTensor(attention_size_total)
            if torch.cuda.is_available():
                pass 
                # indices = indices.cuda()

            # loop over values in attention matrix
            count = 0
            for seq in xrange(attention.size(0)):
                for i in xrange(attention.size(1)):
                    input_index = inputs[seq][i].data[0]
                    # output_index = outputs[seq][output_step].data[0]
                    output_index = outputs[output_step][seq].data[0]

                    # compute corresponding index in confusion matrix
                    confusion_index = output_index*input_vocab_size + input_index
                    indices.index_fill_(0, torch.LongTensor([count]), confusion_index)
                    count+=1

            # add values to confusion matrix
            confusion_matrix.put_(Variable(indices), attention_flat, accumulate=True)
            output_step += 1

        # normalise rows
        confusion_matrix = torch.nn.functional.normalize(confusion_matrix, p=1, dim=1)
        full_confusion_matrix = confusion_matrix

        # Remove rows <unk>, <pad>, <sos>
        row_ids = Variable(torch.LongTensor([2, 3, 4, 5, 6, 7, 8]))
        confusion_matrix = confusion_matrix.index_select(0, row_ids)

        # Only retain columns jump, run, look, walk
        col_ids = Variable(torch.LongTensor([10, 11, 12, 13]))
        confusion_matrix = confusion_matrix.index_select(1, col_ids)

        # compute variance of the confusion matrix:  c1
        variance = torch.sum(torch.var(confusion_matrix, 0))
        if torch.cuda.is_available():
            variance = variance.cuda()

        return variance, full_confusion_matrix