import math
import torch
from torch import nn, Tensor, FloatTensor, LongTensor
from typing import Tuple

from .common import encoder_block, decoder_block
from ..layers import conv2D
from ..ops import select, pool_gru


class ShrinkFork(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, fork_depth: int):
        super(ShrinkFork, self).__init__()
        self.main = encoder_block(in_ch, out_ch)
        aux = [encoder_block(out_ch * 2 ** p, out_ch * 2 ** (p + 1), stride=2)
               for p in range(fork_depth)]
        self.fork = nn.Sequential(*aux)

    def forward(self, x):
        x = self.main(x)
        c = self.fork(x)
        return x, c


class Bilbo(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], enc_depth: int,
                 enc_width: int, fork_depth: int, rnn_width: int):
        super(Bilbo, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")
        max_fork_depth = math.log2(H) - 1
        if fork_depth > max_fork_depth:
            raise AttributeError(
                "fork_depth should be <= {} given the image_size "
                " ({}, {})".format(int(max_fork_depth), H, H))

        self.stem = encoder_block(C, enc_width, stride=1, bn=False)
        encoder_layers = [ShrinkFork(enc_width * 2 ** i, enc_width * 2 ** (i + 1),
                                     fork_depth=(fork_depth - i))
                          for i in range(0, enc_depth - 1)]
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = [decoder_block(enc_width * 2 ** (i + 1), enc_width * 2 ** i)
                          for i in sorted(range(0, enc_depth - 1), reverse=True)]
        last = conv2D(enc_width, C, kernel=3, stride=1)
        self.decoder = nn.Sequential(*decoder_layers, last, nn.Tanh())

        rnn_in = (enc_depth - 1) * enc_width * (H * W / 4) / 2**(fork_depth - 1)
        assert not math.modf(rnn_in)[0]
        rnn_in = int(rnn_in)
        self.gru = nn.GRU(rnn_in, rnn_width, bidirectional=True)
        self.out = nn.Linear(rnn_width * 4, 1, bias=False)

    def forward(self, x: FloatTensor, y: LongTensor) -> Tuple[Tensor, Tensor, Tensor]:
        N, C, D, H, W = x.shape
        hidden, x_rec, features = [], [], []

        for f in range(D):
            cc = []
            h = self.stem(x[:, :, f])
            for i in range(len(self.encoder)):
                h, c = self.encoder[i](h)
                cc.append(c)
            hidden.append(h.unsqueeze(2))
            features.append(torch.cat(cc, dim=1).unsqueeze(2))

            hc = select(h, y)
            x1 = self.decoder(hc)
            x_rec.append(x1.unsqueeze(2))

        hidden = torch.cat(hidden, dim=2)
        x_rec = torch.cat(x_rec, dim=2)
        features = torch.cat(features, dim=2)

        gru_out = self.gru(features.reshape(N, D, -1))
        gru_out = pool_gru(gru_out)

        y_hat = self.out(gru_out)
        return hidden, x_rec, y_hat

    def to_y(self, h: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
