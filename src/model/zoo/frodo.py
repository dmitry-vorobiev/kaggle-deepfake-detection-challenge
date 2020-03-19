import math
import torch

from torch import nn, FloatTensor, LongTensor, Tensor
from typing import List, Tuple

from .common import encoder_block, decoder_block
from ..layers import conv2D
from ..ops import select, decide


def create_enc_blocks(start_width: int, start: int, end: int):
    layers = [encoder_block(start_width * 2**i, start_width * 2**(i+1))
              for i in range(start, end)]
    return layers


def create_dec_blocks(start_width: int, end: int):
    layers = [decoder_block(start_width * 2 ** (i + 1), start_width * 2 ** i)
              for i in sorted(range(0, end), reverse=True)]
    return layers


def reduce_frames(v: List[Tensor]) -> Tensor:
    v = torch.cat(v, dim=2).flatten(2)
    return v.mean(dim=2)


class Frodo(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], enc_depth: int, enc_width: int):
        super(Frodo, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")

        max_depth = math.log2(H) + 1
        if enc_depth > max_depth:
            raise AttributeError(
                "enc_depth should be <= {} given the image_size "
                " ({}, {})".format(int(max_depth), H, H))

        stem = encoder_block(C, enc_width, stride=1, bn=False)
        encoder_layers = create_enc_blocks(enc_width, 0, enc_depth - 1)
        self.encoder = nn.Sequential(stem, *encoder_layers)

        decoder_layers = create_dec_blocks(enc_width, enc_depth - 1)
        last = conv2D(enc_width, C, kernel=3, stride=1)
        self.decoder = nn.Sequential(*decoder_layers, last, nn.Tanh())

    def forward(self, x: FloatTensor, y: LongTensor):
        N, C, D, H, W = x.shape
        hidden, x_rec = [], []

        for f in range(D):
            h = self.encoder(x[:, :, f])
            hc = select(h, y)
            x1 = self.decoder(hc)

            hidden.append(h.unsqueeze(2))
            x_rec.append(x1.unsqueeze(2))

        hidden = torch.cat(hidden, dim=2)
        x_rec = torch.cat(x_rec, dim=2)
        return hidden, x_rec

    def to_y(self, h: FloatTensor, x_rec: FloatTensor):
        return decide(h.detach())


class FrodoV2(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], width: int, enc_depth: int,
                 aux_depth: int, p_drop: float):
        super(FrodoV2, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")

        max_depth = math.log2(H) + 1
        if enc_depth + aux_depth > max_depth:
            raise AttributeError(
                "enc_depth + aux_depth should be <= {} given the "
                "image_size ({}, {})".format(int(max_depth), H, H))

        if width % 2:
            raise AttributeError("width must be even number")

        stem = encoder_block(C, width, stride=1, bn=False)
        encoder_layers = create_enc_blocks(width, 0, enc_depth - 1)
        self.encoder = nn.Sequential(stem, *encoder_layers)

        decoder_layers = create_dec_blocks(width, enc_depth - 1)
        dec_out = conv2D(width, C, kernel=3, stride=1)
        self.decoder = nn.Sequential(*decoder_layers, dec_out, nn.Tanh())

        for i in range(2):
            aux = create_enc_blocks(width // 2, enc_depth - 1, enc_depth - 1 + aux_depth)
            setattr(self, 'aux_{}'.format(i), nn.Sequential(*aux))

        out_dim = width * 2 ** (enc_depth + aux_depth - 1)
        self.aux_out = nn.Sequential(
            nn.Dropout(p=p_drop),
            nn.Linear(out_dim, 1, bias=False))

    def forward(self, x: FloatTensor, y: LongTensor):
        N, C, D, H, W = x.shape
        enc, x_rec, aux_0, aux_1 = [], [], [], []

        for f in range(D):
            h = self.encoder(x[:, :, f])
            hc = select(h, y)
            x1 = self.decoder(hc)

            h0, h1 = torch.chunk(h, 2, dim=1)
            a0 = self.aux_0(h0)
            a1 = self.aux_1(h1)

            for val, arr in zip([h, x1, a0, a1],
                                [enc, x_rec, aux_0, aux_1]):
                val = val.unsqueeze(2)
                arr.append(val)

        enc = torch.cat(enc, dim=2)
        x_rec = torch.cat(x_rec, dim=2)
        aux_0 = reduce_frames(aux_0)
        aux_1 = reduce_frames(aux_1)
        aux = torch.cat([aux_0, aux_1], dim=1)
        y_hat = self.aux_out(aux)
        return enc, x_rec, y_hat

    def to_y(self, enc: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
