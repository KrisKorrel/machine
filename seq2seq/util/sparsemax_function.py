"""Sparsemax activation function.

From https://github.com/vene/sparse-structured-attention
Edited slightly because it did not work out of the box.
"""

from __future__ import division

import torch
import torch.autograd as ta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SparsemaxFunction(ta.Function):
    """Applies a sample-wise normalizing projection over a batch."""

    def forward(self, x, lengths=None):
        """Forward activation."""
        requires_squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            requires_squeeze = True

        n_samples, max_dim = x.size()

        has_lengths = True
        if lengths is None:
            has_lengths = False
            lengths = [max_dim] * n_samples

        y_star = x.new()
        y_star.resize_as_(x)
        y_star.zero_()

        for i in range(n_samples):
            y_star[i, :lengths[i]] = self.project(x[i, :lengths[i]])

        if requires_squeeze:
            y_star = y_star.squeeze()

        if has_lengths:
            self.save_for_backward(y_star, lengths)
        else:
            self.save_for_backward(y_star)

        return y_star

    def backward(self, dout):
        """Backward propagation."""
        if not self.needs_input_grad[0]:
            return None

        if len(self.needs_input_grad) > 1 and self.needs_input_grad[1]:
            raise ValueError("Cannot differentiate {} w.r.t. the "
                             "sequence lengths".format(self.__name__))

        saved = self.saved_tensors
        if len(saved) == 2:
            y_star, lengths = saved
        else:
            y_star, = saved
            lengths = None

        requires_squeeze = False
        if y_star.dim() == 1:
            y_star = y_star.unsqueeze(0)
            dout = dout.unsqueeze(0)
            requires_squeeze = True

        n_samples, max_dim = y_star.size()
        din = dout.new()
        din.resize_as_(y_star)
        din.zero_()

        if lengths is None:
            lengths = [max_dim] * n_samples

        for i in range(n_samples):
            din[i, :lengths[i]] = self.project_jv(dout[i, :lengths[i]],
                                                  y_star[i, :lengths[i]])

        if requires_squeeze:
            din = din.squeeze()

        return din, None

    def project(self, v, z=1):
        """Project."""
        v_sorted, _ = torch.sort(v, dim=0, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - z
        ind = torch.arange(1, 1 + len(v), device=device)
        cond = v_sorted - cssv / ind > 0
        rho = ind.masked_select(cond)[-1]
        tau = cssv.masked_select(cond)[-1] / rho
        w = torch.clamp(v - tau, min=0)
        return w

    def project_jv(self, dout, w_star):
        """Project jv."""
        supp = w_star > 0
        masked = dout.masked_select(supp)
        masked -= masked.sum() / supp.sum()
        out = dout.new(dout.size()).zero_()
        out[supp] = masked
        return(out)
