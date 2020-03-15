import torch
from torch import nn, Tensor
from typing import Callable


def conv2D(in_ch: int, out_ch: int, kernel=3, stride=1, pad=1, bias=True, a=0) -> nn.Conv2d:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                     padding=pad, bias=bias)
    nn.init.kaiming_normal_(conv.weight, a=a)
    if bias:
        conv.bias.data.zero_()
    return conv


def conv3D(in_ch: int, out_ch: int, kernel=3, stride=1, pad=1, bias=True, a=0) -> nn.Conv3d:
    conv = nn.Conv3d(in_ch, out_ch,
                     kernel_size=(1, kernel, kernel),
                     stride=(1, stride, stride),
                     padding=(0, pad, pad),
                     bias=bias)
    nn.init.kaiming_normal_(conv.weight, a=a)
    if bias:
        conv.bias.data.zero_()
    return conv


class Lambda(nn.Module):
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        super(Lambda, self).__init__()
        self.fn = fn
        
    def forward(self, x: Tensor):
        return self.fn(x)
