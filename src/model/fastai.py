"""
https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/text/models/awd_lstm.py#L17
"""
from typing import Collection
from torch import nn, Tensor


def dropout_mask(x: Tensor, sz: Collection[int], p: float):
    """Return a dropout mask of the same type as `x`, " \
    "size `sz`, with probability `p` to cancel an element."""
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    """Dropout with probability `p` that is consistent on the seq_len dimension."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m
