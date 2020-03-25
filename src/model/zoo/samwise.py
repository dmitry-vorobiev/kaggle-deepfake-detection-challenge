import math
import torch

from torch import nn, FloatTensor, LongTensor, Tensor
from typing import Optional, Tuple

from ..layers import conv2D, Lambda, MaxMean3D, EncoderBlock, DecoderBlock
from ..ops import select


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


class RNNBlock(nn.Module):
    def __init__(self, in_ch: int, rnn_ch: int, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(in_ch, rnn_ch, bidirectional=bidirectional)
        self.out_ch = rnn_ch * 3 * (2 if bidirectional else 1)

    def forward(self, x):
        N, C, D, H, W = x.shape
        x = x.reshape(N, D, -1)
        # N, D, C -> D, N, C
        x = x.transpose(0, 1)
        x, _ = self.gru(x)
        x_mean = x.mean(0)
        x_max, _ = x.max(0)
        x_last = x[-1]
        x = torch.cat([x_mean, x_max, x_last], dim=1)
        return x


class Samwise(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], width: int,
                 enc_depth: int, aux_depth: int, wide=False,
                 reduce: str = 'mean', rnn_dim: Optional[int] = None,
                 p_emb_drop=0.1, p_out_drop=0.1, train=True):
        super(Samwise, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")

        max_depth = math.log2(H)
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
            if p_emb_drop > 0:
                aux_branch = [nn.Dropout2d(p=p_emb_drop)] + aux_branch
            setattr(self, f'aux_{i}', nn.Sequential(*aux_branch))

        aux_dim = width * 2 ** (enc_depth + aux_depth)
        out_dim = 0
        if reduce == 'rnn':
            if not rnn_dim:
                raise AttributeError("GRU dim is missing")
            for i in range(2):
                reducer = RNNBlock(aux_dim, rnn_dim, bidirectional=True)
                out_dim = reducer.out_ch * 2
                setattr(self, f'reduce_{i}', reducer)
        elif reduce == 'mean':
            reducer = Lambda(lambda x: x.mean(dim=(2, 3, 4)))
            out_dim = aux_dim // 2
            for i in range(2):
                setattr(self, f'reduce_{i}', reducer)
        elif reduce == 'max-mean' or reduce == 'mean-max':
            reducer = MaxMean3D()
            out_dim = aux_dim
            for i in range(2):
                setattr(self, f'reduce_{i}', reducer)
        else:
            raise AttributeError(
                f"reduce={reduce} - invalid value, available options: "
                "[rnn, mean, max-mean, mean-max]")

        aux_out = [nn.Linear(out_dim, 1, bias=False)]
        if p_out_drop > 0:
            aux_out = [nn.Dropout(p=p_out_drop)] + aux_out
        self.aux_out = nn.Sequential(*aux_out)
        self.is_train = train

    def forward(self, x: FloatTensor, y: Optional[LongTensor] = None):
        N, C, D, H, W = x.shape
        hidden, x_rec, aux_0, aux_1 = [], [], [], []

        for f in range(D):
            h = self.encoder(x[:, :, f])

            if self.is_train:
                hc = select(h, y)
                x1 = self.decoder(hc).unsqueeze(2)
                x_rec.append(x1)

            h0, h1 = torch.chunk(h, 2, dim=1)
            a0 = self.aux_0(h0)
            a1 = self.aux_1(h1)

            for val, arr in zip([h, a0, a1], [hidden, aux_0, aux_1]):
                val = val.unsqueeze(2)
                arr.append(val)

        hidden = torch.cat(hidden, dim=2)
        x_rec = torch.cat(x_rec, dim=2) if self.is_train else None

        aux_out = []
        for i, aux_i in enumerate([aux_0, aux_1]):
            aux_i = torch.cat(aux_i, dim=2)
            aux_i = getattr(self, f'reduce_{i}')(aux_i)
            aux_out.append(aux_i)
        aux_out = torch.cat(aux_out, dim=1)

        y_hat = self.aux_out(aux_out)
        return hidden, x_rec, y_hat

    @staticmethod
    def to_y(enc: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
