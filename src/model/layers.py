import torch
from torch import nn, Tensor
from typing import Callable


def conv3D(in_ch: int, out_ch: int, kernel=3, stride=1, pad=1) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, 
                     kernel_size=(1, kernel, kernel), 
                     stride=(1, stride, stride), 
                     padding=(0, pad, pad))


class Lambda(nn.Module):
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        super(Lambda, self).__init__()
        self.fn = fn
        
    def forward(self, x: Tensor):
        return self.fn(x)
