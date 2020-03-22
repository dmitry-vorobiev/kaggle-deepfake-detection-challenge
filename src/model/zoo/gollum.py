import torch
from torch import nn, FloatTensor, LongTensor, Tensor
from typing import List, Tuple

from .common import encoder_block, decoder_block
from ..layers import conv2D, conv3D, Lambda
from ..ops import identity, pool_gru, select

DetectorOut = Tuple[Tensor, Tensor, Tensor]


def middle_block(in_ch: int, out_ch: int, kernel=3, stride=2, bn=True) -> nn.Module:
    conv = conv3D(in_ch, out_ch, kernel=kernel, stride=stride, bias=not bn)
    relu = nn.ReLU(inplace=True)
    layers = [conv, relu]
    if bn:
        layers.append(nn.BatchNorm3d(out_ch))
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

    def forward(self, x: FloatTensor, y: LongTensor) -> Tuple[FloatTensor, FloatTensor]:
        h = self.encoder(x)
        hc = select(h, y)
        x_hat = self.decoder(hc)
        return h, x_hat


class Gollum(nn.Module):
    def __init__(self, img_size: int, enc_depth: int, enc_width: int,
                 mid_layers: List[int], out_ch: int,
                 pool_size: Tuple[int, int] = None):
        super(Gollum, self).__init__()
        if img_size % 32:
            raise AttributeError("img_size should be a multiple of 32")
        if out_ch % 2:
            raise AttributeError("out_ch should be an even number")

        size_factor = 2 ** (enc_depth - 1)
        if size_factor > img_size:
            raise AttributeError(
                'Encoder dims (%d, %d) are incompatible with image '
                'size (%d, %d)' % (enc_depth, enc_width, img_size, img_size))
        emb_size = img_size // size_factor
        emb_ch = enc_width * size_factor

        self.encoder = AutoEncoder(in_ch=3, depth=enc_depth, width=enc_width)

        if img_size // 2 ** (enc_depth - 1) == 1:
            self.middle = Lambda(identity)
            rnn_in = emb_ch

        elif len(mid_layers) > 0:
            n_mid = len(mid_layers)
            mid_layers = [emb_ch] + list(mid_layers)
            out_size = emb_size // 2 ** n_mid
            if not out_size:
                raise AssertionError('Too many middle layers...')
            layers = [middle_block(mid_layers[i], mid_layers[i + 1], stride=2)
                      for i in range(n_mid)]
            self.middle = nn.Sequential(*layers)
            rnn_in = mid_layers[-1] * out_size ** 2

        elif pool_size is not None:
            D, H = pool_size
            self.middle = nn.AdaptiveAvgPool3d((D, H, H))
            rnn_in = emb_ch * H ** 2

        else:
            raise AttributeError(
                'Both mid_layers and pool_size are missing. '
                'Unable to build model with provided configuration')

        self.rnn = nn.GRU(rnn_in, out_ch // 2)
        self.out = nn.Linear(out_ch, 1, bias=False)

    def forward(self, x: FloatTensor, y: LongTensor) -> DetectorOut:
        N, C, D, H, W = x.shape
        hidden, xs_hat = [], []

        for f in range(D):
            h, x_hat = self.encoder(x[:, :, f], y)
            hidden.append(h.unsqueeze(2))
            xs_hat.append(x_hat.unsqueeze(2))

        hidden = torch.cat(hidden, dim=2)
        xs_hat = torch.cat(xs_hat, dim=2)

        seq = self.middle(hidden).reshape(N, D, -1).transpose(0, 1)
        seq_out = self.rnn(seq)
        seq_out = pool_gru(seq_out)
        y_hat = self.out(seq_out)

        return hidden, xs_hat, y_hat

    @staticmethod
    def to_y(h: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
