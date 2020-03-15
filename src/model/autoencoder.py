
import torch.nn.functional as F
from functools import partial
from torch import nn, Tensor, FloatTensor, LongTensor
from typing import Tuple

from .layers import conv2D, Lambda
from .ops import image_grad, select

AutoEncoderOut = Tuple[FloatTensor, FloatTensor]


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


class AutoEncoder(nn.Module):
    def __init__(self, in_ch: int, depth: int, width: int):
        super(AutoEncoder, self).__init__()
        self.encoder = AutoEncoder._build_encoder(in_ch, depth, width)
        self.decoder = AutoEncoder._build_decoder(in_ch, depth, width)

    @staticmethod
    def _build_encoder(in_ch: int, depth: int, size: int) -> nn.Module:
        stem = encoder_block(in_ch, size, stride=1, bn=False)
        main = [encoder_block(size * 2**i, size * 2**(i+1))
                for i in range(0, depth - 1)]
        return nn.Sequential(stem, *main)

    @staticmethod
    def _build_decoder(out_ch: int, depth: int, size: int) -> nn.Module:
        main = [decoder_block(size * 2**(i+1), size * 2**i)
                for i in sorted(range(0, depth - 1), reverse=True)]
        last = conv2D(size, out_ch, kernel=3, stride=1)
        return nn.Sequential(*main, last, nn.Tanh())

    def forward(self, x: FloatTensor, y: LongTensor) -> AutoEncoderOut:
        x = image_grad(x, n=3, keep_size=True)
        h = self.encoder(x)
        hc = select(h, y)
        x_hat = self.decoder(hc)
        return h, x_hat
