
import torch.nn.functional as F
from functools import partial
from torch import nn, Tensor, FloatTensor, LongTensor
from typing import Tuple

from .layers import conv2D, Lambda
from .ops import image_grad, select

AutoEncoderOut = Tuple[FloatTensor, FloatTensor]


def encoder_block(in_ch: int, out_ch: int, kernel=3, stride=2, pad=0, bn=True) -> nn.Module:
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=stride, pad=pad, a=0)
    relu = nn.ReLU(inplace=True)
    if bn:
        layers = [conv, nn.BatchNorm2d(out_ch), relu]
    else:
        layers = [conv, relu]
    return nn.Sequential(*layers)


def decoder_block(in_ch: int, out_ch: int, kernel=3, scale=2, pad=0, bn=True) -> nn.Module:
    upsample = Lambda(partial(F.interpolate, scale_factor=scale, mode='nearest'))
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=1, pad=pad, a=0)
    relu = nn.ReLU(inplace=True)
    if bn:
        layers = [upsample, conv, nn.BatchNorm2d(out_ch), relu]
    else:
        layers = [upsample, conv, relu]
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, in_ch: int, depth: int, size=8, pad=1):
        super(AutoEncoder, self).__init__()
        self.encoder = AutoEncoder._build_encoder(in_ch, depth, size, pad)
        self.decoder = AutoEncoder._build_decoder(in_ch, depth, size, pad)
        
    @staticmethod
    def _build_encoder(in_ch: int, depth: int, size: int, pad: int) -> nn.Module:
        stem = encoder_block(in_ch, size, stride=1, pad=pad, bn=False)
        main = [encoder_block(size * 2**i, size * 2**(i+1), pad=pad)
                for i in range(0, depth - 1)]
        return nn.Sequential(stem, *main)

    @staticmethod
    def _build_decoder(out_ch: int, depth: int, size: int, pad: int) -> nn.Module:
        main = [decoder_block(size * 2**(i+1), size * 2**i, pad=pad)
                for i in sorted(range(0, depth - 1), reverse=True)]
        last = conv2D(size, out_ch, kernel=3, stride=1, pad=pad, a=1)
        return nn.Sequential(*main, last, nn.Tanh())

    def forward(self, x: FloatTensor, y: LongTensor) -> AutoEncoderOut:
        x = image_grad(x, n=3, keep_size=True)
        h = self.encoder(x)
        hc = select(h, y)
        x_hat = self.decoder(hc)
        return h, x_hat
