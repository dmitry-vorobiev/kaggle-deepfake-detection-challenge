from functools import partial

import re
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Callable, Union, Optional

from torch.nn import functional as F

relu = nn.ReLU(inplace=True)


def is_torch_version_ge_1_6():
    m = re.match(r"(\d)\.(\d)", torch.__version__)
    if len(m.groups()) == 2:
        major = m.group(1)
        minor = m.group(2)
        if int(major) >= 1 and int(minor) >= 6:
            return True
    return False


def get_a_from_act_fn(func: Optional[nn.Module] = None) -> float:
    if func is None:
        a = 1.0
    elif isinstance(func, nn.ReLU):
        a = 0.0
    elif isinstance(func, nn.PReLU):
        # can't pass PReLU with per-channel parameters from config
        assert func.num_parameters == 1
        a = func.weight.item()
    elif isinstance(func, nn.LeakyReLU):
        a = func.negative_slope
    else:
        raise NotImplementedError()
    return a


def conv2D(in_ch: int, out_ch: int, kernel=3, stride=1, bias=False, a=0.) -> nn.Conv2d:
    pad = kernel // 2
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                     padding=pad, bias=bias)
    nn.init.kaiming_normal_(conv.weight, a=a)
    if bias:
        nn.init.constant_(conv.bias, 0)
    return conv


def conv3D(in_ch: int, out_ch: int, kernel=3, stride=1, bias=False, a=0.) -> nn.Conv3d:
    pad = kernel // 2
    conv = nn.Conv3d(in_ch, out_ch,
                     kernel_size=(1, kernel, kernel),
                     stride=(1, stride, stride),
                     padding=(0, pad, pad),
                     bias=bias)
    nn.init.kaiming_normal_(conv.weight, a=a)
    if bias:
        nn.init.constant_(conv.bias, 0)
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


ActivationFn = Union[nn.Module, Callable[[Tensor], Tensor]]


def enc_layer(in_ch: int, out_ch: int, kernel=3, stride=1,
              act_fn: Optional[ActivationFn] = relu,
              zero_bn=False) -> nn.Module:
    a = get_a_from_act_fn(act_fn)
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=stride, bias=False, a=a)
    bn = nn.BatchNorm2d(out_ch)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv, bn]
    if act_fn is not None:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hd_ch: int,
                 act_fn: Optional[ActivationFn] = relu):
        super().__init__()
        self.conv = nn.Sequential(
            enc_layer(in_ch, hd_ch,  kernel=1, act_fn=act_fn),
            enc_layer(hd_ch, hd_ch,  kernel=3, stride=2, act_fn=act_fn),
            enc_layer(hd_ch, out_ch, kernel=1, zero_bn=True, act_fn=None))
        self.id_conv = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            enc_layer(in_ch, out_ch, kernel=1, act_fn=None))
        self.act_fn = act_fn

    def forward(self, x):
        x = self.conv(x) + self.id_conv(x)
        return self.act_fn(x)


def upscale(scale: int):
    """
    /anaconda3/envs/torch-xla-nightly/lib/python3.6/site-packages/torch/nn/functional.py:2875:
    UserWarning: The default behavior for interpolate/upsample with float scale_factor
    will change in 1.6.0 to align with other frameworks/libraries, and use scale_factor directly,
    instead of relying on the computed output size. If you wish to keep the old behavior,
    please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.
    """
    kwargs = dict()
    if is_torch_version_ge_1_6():
        kwargs = dict(recompute_scale_factor=True)
    return Lambda(partial(F.interpolate, scale_factor=scale, mode='nearest', **kwargs))


def dec_layer(in_ch: int, out_ch: int, kernel=3, scale=1,
              act_fn: Optional[ActivationFn] = relu, zero_bn=False) -> nn.Module:
    layers = [upscale(scale)] if scale > 1 else []
    a = get_a_from_act_fn(act_fn)
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=1, bias=False, a=a)
    bn = nn.BatchNorm2d(out_ch)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers += [conv, bn]
    if act_fn is not None:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hd_ch: int,
                 act_fn: Optional[ActivationFn] = relu):
        super().__init__()
        self.conv = nn.Sequential(
            dec_layer(in_ch, hd_ch,  kernel=1, act_fn=act_fn),
            dec_layer(hd_ch, hd_ch,  kernel=3, scale=2, act_fn=act_fn),
            dec_layer(hd_ch, out_ch, kernel=1, zero_bn=True, act_fn=None))
        self.id_conv = dec_layer(in_ch, out_ch, kernel=1, scale=2, act_fn=None)
        self.act_fn = act_fn

    def forward(self, x):
        x = self.conv(x) + self.id_conv(x)
        return self.act_fn(x)


class MaxMean2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x_mean = F.avg_pool2d(x, (H, W))
        x_max = F.max_pool2d(x, (H, W))
        x = torch.cat([x_mean, x_max], dim=1)
        return x.reshape(N, -1)


class MaxMean3D(nn.Module):
    def __init__(self, reduce_frames=True):
        super().__init__()
        self.reduce_frames = reduce_frames

    def forward(self, x):
        N, C, D, H, W = x.shape
        kernel = (D, H, W) if self.reduce_frames else (1, H, W)
        x_mean = F.avg_pool3d(x, kernel)
        x_max = F.max_pool3d(x, kernel)
        x = torch.cat([x_mean, x_max], dim=1)
        return x.reshape(N, -1) if self.reduce_frames else x.reshape(N, -1, D)
