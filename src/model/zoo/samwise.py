import math
import torch
import torch.nn.functional as F

from functools import partial
from torch import nn, FloatTensor, LongTensor, Tensor
from typing import Callable, Optional, Tuple, Union

from .frodo import reduce_frames
from ..layers import conv2D, Lambda
from ..ops import select


ActivationFn = Union[nn.Module, Callable[[Tensor], Tensor]]


def enc_layer(in_ch: int, out_ch: int, kernel=3, stride=1,
              act_fn: Optional[ActivationFn] = nn.ReLU(inplace=True),
              zero_bn=False) -> nn.Module:
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=stride, bias=False)
    bn = nn.BatchNorm2d(out_ch)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv, bn]
    if act_fn is not None:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hd_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            enc_layer(in_ch, hd_ch,  kernel=1),
            enc_layer(hd_ch, hd_ch,  kernel=3, stride=2),
            enc_layer(hd_ch, out_ch, kernel=1, zero_bn=True, act_fn=None))
        self.id_conv = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            enc_layer(in_ch, out_ch, kernel=1, act_fn=None))

    def forward(self, x):
        x = self.conv(x) + self.id_conv(x)
        return torch.relu_(x)


def upscale(scale: int):
    return Lambda(partial(F.interpolate, scale_factor=scale, mode='nearest'))


def dec_layer(in_ch: int, out_ch: int, kernel=3, scale=1,
              act_fn: Optional[ActivationFn] = nn.ReLU(inplace=True),
              zero_bn=False) -> nn.Module:
    layers = [upscale(scale)] if scale > 1 else []
    conv = conv2D(in_ch, out_ch, kernel=kernel, stride=1, bias=False)
    bn = nn.BatchNorm2d(out_ch)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers += [conv, bn]
    if act_fn is not None:
        layers.append(act_fn)
    return nn.Sequential(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hd_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            dec_layer(in_ch, hd_ch,  kernel=1),
            dec_layer(hd_ch, hd_ch,  kernel=3, scale=2),
            dec_layer(hd_ch, out_ch, kernel=1, zero_bn=True, act_fn=None))
        self.id_conv = dec_layer(in_ch, out_ch, kernel=1, scale=2, act_fn=None)

    def forward(self, x):
        x = self.conv(x) + self.id_conv(x)
        return torch.relu_(x)


def stack_enc_blocks(width: int, start: int, end: int, wide=False):
    layers = []
    for i in range(start, end):
        in_ch = width * 2**i
        out_ch = width * 2**(i+1)
        h_ch = out_ch if wide else in_ch
        block = EncoderBlock(in_ch, out_ch, h_ch)
        layers.append(block)
    return layers


def stack_dec_blocks(width: int, start: int, end: int, wide=False):
    layers = []
    for i in sorted(range(start, end), reverse=True):
        in_ch = width * 2**(i+1)
        out_ch = width * 2**i
        h_ch = in_ch if wide else out_ch
        block = DecoderBlock(in_ch, out_ch, h_ch)
        layers.append(block)
    return layers


class Samwise(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], width: int,
                 enc_depth: int, aux_depth: int, p_drop=0.0, wide=False):
        super(Samwise, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")

        max_depth = math.log2(H) + 1
        if enc_depth + aux_depth > max_depth:
            raise AttributeError(
                f"enc_depth + aux_depth should be <= {int(max_depth)} given the "
                f"image_size ({H}, {H})")

        if width % 2:
            raise AttributeError("width must be even number")

        stem = [conv2D(C, width, bias=False), nn.ReLU(inplace=True)]
        encoder = stack_enc_blocks(width, 0, enc_depth - 1, wide=wide)
        self.encoder = nn.Sequential(*stem, *encoder)

        decoder = stack_dec_blocks(width, 0, enc_depth - 1, wide=wide)
        dec_out = conv2D(width, C, kernel=3, stride=1, bias=False)
        self.decoder = nn.Sequential(*decoder, dec_out, nn.Tanh())

        for i in range(2):
            aux_branch = stack_enc_blocks(
                width // 2, enc_depth - 1, enc_depth - 1 + aux_depth, wide=wide)
            setattr(self, 'aux_{}'.format(i), nn.Sequential(*aux_branch))

        out_dim = width * 2 ** (enc_depth + aux_depth - 1)
        self.aux_out = nn.Sequential(
            nn.Dropout(p=p_drop),
            nn.Linear(out_dim, 1, bias=False))

    def forward(self, x: FloatTensor, y: LongTensor):
        N, C, D, H, W = x.shape
        hidden, x_rec, aux_0, aux_1 = [], [], [], []

        for f in range(D):
            h = self.encoder(x[:, :, f])
            hc = select(h, y)
            x1 = self.decoder(hc)

            h0, h1 = torch.chunk(h, 2, dim=1)
            a0 = self.aux_0(h0)
            a1 = self.aux_1(h1)

            for val, arr in zip([h, x1, a0, a1],
                                [hidden, x_rec, aux_0, aux_1]):
                val = val.unsqueeze(2)
                arr.append(val)

        hidden = torch.cat(hidden, dim=2)
        x_rec = torch.cat(x_rec, dim=2)
        aux_0 = reduce_frames(aux_0)
        aux_1 = reduce_frames(aux_1)
        aux = torch.cat([aux_0, aux_1], dim=1)
        y_hat = self.aux_out(aux)
        return hidden, x_rec, y_hat

    @staticmethod
    def to_y(enc: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
