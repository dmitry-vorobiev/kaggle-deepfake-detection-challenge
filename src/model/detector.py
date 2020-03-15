import torch
from torch import nn, FloatTensor, LongTensor, Tensor
from typing import List, Tuple

from .autoencoder import AutoEncoder
from .layers import conv3D, Lambda
from .ops import identity, pool_gru

DetectorOut = Tuple[Tensor, Tensor, Tensor]


def intermediate_block(in_ch: int, out_ch: int, stride=2) -> nn.Module:
    layers = nn.Sequential(
        conv3D(in_ch, out_ch, kernel=3, stride=stride, pad=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )
    return layers


class FakeDetector(nn.Module):
    def __init__(self, img_size: int, enc_depth: int, enc_width: int,
                 mid_layers: List[int], out_ch: int,
                 pool_size: Tuple[int, int] = None):
        super(FakeDetector, self).__init__()
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

        self.encoder = AutoEncoder(in_ch=3, depth=enc_depth, size=enc_width, pad=1)

        if img_size // 2 ** (enc_depth - 1) == 1:
            self.middle = Lambda(identity)
            rnn_in = emb_ch

        elif len(mid_layers) > 0:
            n_mid = len(mid_layers)
            mid_layers = [emb_ch] + mid_layers
            out_size = emb_size // 2 ** n_mid
            if not out_size:
                raise AssertionError('Too many middle layers...')
            layers = [intermediate_block(mid_layers[i], mid_layers[i + 1], stride=2)
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

        seq = self.middle(hidden).reshape(N, D, -1)
        seq_out = self.rnn(seq)
        seq_out = pool_gru(seq_out)
        y_hat = self.out(seq_out)

        return hidden, xs_hat, y_hat


def basic_detector_256():
    return FakeDetector(img_size=256, enc_depth=5, enc_width=8, mid_layers=[256, 256], out_ch=128)

