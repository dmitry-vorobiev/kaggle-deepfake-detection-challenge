import math
import torch

from torch import nn, FloatTensor, LongTensor, Tensor
from typing import Tuple

from .common import encoder_block, decoder_block
from ..layers import conv2D
from ..ops import select, decide


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
        encoder_layers = [encoder_block(enc_width * 2**i, enc_width * 2**(i+1))
                          for i in range(0, enc_depth - 1)]
        self.encoder = nn.Sequential(stem, *encoder_layers)

        decoder_layers = [decoder_block(enc_width * 2**(i+1), enc_width * 2**i)
                          for i in sorted(range(0, enc_depth - 1), reverse=True)]
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
    def __init__(self, image_shape: Tuple[int, int, int],
                 enc_depth: int, enc_width: int, aux_depth: int):
        super(FrodoV2, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")

        max_depth = math.log2(H) + 1
        if enc_depth + aux_depth > max_depth:
            raise AttributeError(
                "enc_depth + aux_depth should be <= {} given the "
                "image_size ({}, {})".format(int(max_depth), H, H))

        stem = encoder_block(C, enc_width, stride=1, bn=False)
        encoder_layers = [encoder_block(enc_width * 2 ** i, enc_width * 2 ** (i + 1))
                          for i in range(0, enc_depth - 1)]
        self.encoder = nn.Sequential(stem, *encoder_layers)

        aux_layers = [encoder_block(enc_width * 2 ** i, enc_width * 2 ** (i + 1))
                      for i in range(enc_depth - 1, enc_depth + aux_depth - 1)]
        self.aux = nn.Sequential(*aux_layers)
        aux_out_dim = enc_width * 2 ** (enc_depth + aux_depth)
        self.aux_out = nn.Linear(aux_out_dim, 1, bias=False)

        decoder_layers = [decoder_block(enc_width * 2 ** (i + 1), enc_width * 2 ** i)
                          for i in sorted(range(0, enc_depth - 1), reverse=True)]
        last = conv2D(enc_width, C, kernel=3, stride=1)
        self.decoder = nn.Sequential(*decoder_layers, last, nn.Tanh())

    def forward(self, x: FloatTensor, y: LongTensor):
        N, C, D, H, W = x.shape
        hidden, x_rec, auxs = [], [], []

        for f in range(D):
            h = self.encoder(x[:, :, f])
            hc = select(h, y)
            x1 = self.decoder(hc)
            a = self.aux(h)

            hidden.append(h.unsqueeze(2))
            x_rec.append(x1.unsqueeze(2))
            auxs.append(a.unsqueeze(2))

        hidden = torch.cat(hidden, dim=2)
        x_rec = torch.cat(x_rec, dim=2)
        auxs = torch.cat(auxs, dim=2).flatten(2)
        a_max, _ = auxs.max(dim=2)
        a_mean = auxs.mean(dim=2)
        auxs = torch.cat([a_max, a_mean], dim=1)
        y_hat = self.aux_out(auxs)
        return hidden, x_rec, y_hat

    def to_y(self, h: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
