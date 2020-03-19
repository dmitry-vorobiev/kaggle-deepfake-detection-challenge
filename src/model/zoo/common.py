
import torch.nn.functional as F
from functools import partial
from torch import nn, FloatTensor
from typing import Tuple

from ..layers import conv2D, Lambda

ModelOut = Tuple[FloatTensor, FloatTensor, FloatTensor]


def encoder_block(in_ch: int, out_ch: int, kernel=3, stride=2, bn=True) -> nn.Module:
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=stride, bias=not bn)
    relu = nn.ReLU(inplace=True)
    layers = [conv, relu]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers)


def decoder_block(in_ch: int, out_ch: int, kernel=3, scale=2, bn=True) -> nn.Module:
    upsample = Lambda(partial(F.interpolate, scale_factor=scale, mode='nearest'))
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=1, bias=not bn)
    relu = nn.ReLU(inplace=True)
    layers = [upsample, conv, relu]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers)
