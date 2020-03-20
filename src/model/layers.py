import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Callable


def conv2D(in_ch: int, out_ch: int, kernel=3, stride=1, bias=True, a=0) -> nn.Conv2d:
    pad = kernel // 2
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                     padding=pad, bias=bias)
    nn.init.kaiming_normal_(conv.weight, a=a)
    if bias:
        conv.bias.data.zero_()
    return conv


def conv3D(in_ch: int, out_ch: int, kernel=3, stride=1, bias=True, a=0) -> nn.Conv3d:
    pad = kernel // 2
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
    def __init__(self, fn: Callable[[Any], Tensor]):
        super(Lambda, self).__init__()
        self.fn = fn
        
    def forward(self, x: Tensor):
        return self.fn(x)


def affine(ch_in: int, ch_out: int):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False)


class SpatialAttention(nn.Module):
    """
    Original code:
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
    """
    def __init__(self, ch):
        super(SpatialAttention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = affine(ch, ch // 8)
        self.phi = affine(ch, ch // 8)
        self.g = affine(ch, ch // 2)
        self.o = affine(ch // 2, ch)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        N, C, H, W = x.shape
        ch = self.ch

        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, ch // 8, H * W)
        phi = phi.view(-1, ch // 8, H * W // 4)
        g = g.view(-1, ch // 2, H * W // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, ch // 2, H, W))
        return self.gamma * o + x
