"""Sparsemax activation function wrapper.

From https://github.com/vene/sparse-structured-attention
Edited slightly because it did not work out of the box.

SparsemaxFunction is the actual function. Sparsemax is the module that calls the function.
Otherwise you would have to call sparsemax.forward(x) instead of sparsemax(x)
"""

from .sparsemax_function import SparsemaxFunction

import torch.nn as nn


class Sparsemax(nn.Module):
    """Sparsemax module."""

    def __init__(self):
        """Initialize sparsemax function."""
        super(Sparsemax, self).__init__()
        self.sparsemax_function = SparsemaxFunction()

    def forward(self, x, lengths=None):
        """Forward progration."""
        return self.sparsemax_function.forward(x, lengths)
