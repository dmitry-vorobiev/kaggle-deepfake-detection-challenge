import math
import torch

from torch import nn, FloatTensor, LongTensor, Tensor
from typing import List, Optional, Tuple

from ..efficient_attention.efficient_attention import EfficientAttention
from ..layers import conv2D, Lambda, MaxMean2D, EncoderBlock, DecoderBlock
from ..ops import select


class RNNBlock(nn.Module):
    def __init__(self, in_ch: int, rnn_ch: int, cls=nn.GRU, bidirectional=False):
        super().__init__()
        self.rnn = cls(in_ch, rnn_ch, bidirectional=bidirectional)
        self.out_ch = rnn_ch * 2 * (2 if bidirectional else 1)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, h = self.rnn(x)
        x_max, _ = x.max(0)
        x_mean = x.mean(0)
        x = torch.cat([x_max, x_mean], dim=1)
        return x


def stack_enc_blocks(width: int, start: int, end: int, wide=False,
                     attention: Optional[List[int]] = None):
    if attention is None:
        attention = []
    layers = []
    for i in range(start, end):
        in_ch = width * 2**i
        out_ch = width * 2**(i+1)
        if i in attention:
            assert not in_ch % 4
            att = EfficientAttention(in_ch, in_ch, in_ch//4, in_ch)
            layers.append(att)
        h_ch = out_ch if wide else in_ch
        block = EncoderBlock(in_ch, out_ch, h_ch)
        layers.append(block)
    return layers


def stack_dec_blocks(width: int, start: int, end: int, wide=False,
                     attention: Optional[List[int]] = None):
    layers = []
    for i in sorted(range(start, end), reverse=True):
        in_ch = width * 2**(i+1)
        out_ch = width * 2**i
        h_ch = in_ch if wide else out_ch
        block = DecoderBlock(in_ch, out_ch, h_ch)
        layers.append(block)
        if i in attention:
            assert not out_ch % 4
            att = EfficientAttention(out_ch, out_ch, out_ch//4, out_ch)
            layers.append(att)
    return layers


class SenyaGanjubas(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], width: int,
                 enc_depth: int, aux_depth: int, rnn_dim: int, wide=False,
                 p_emb_drop=0.1, p_out_drop=0.1, train=True,
                 enc_att: Optional[List[int]] = None,
                 dec_att: Optional[List[int]] = None,
                 aux_att: Optional[List[int]] = None):
        super(SenyaGanjubas, self).__init__()
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
        encoder = stack_enc_blocks(width, 0, enc_depth - 1, wide=wide, attention=enc_att)
        self.encoder = nn.Sequential(*stem, *encoder)

        decoder = stack_dec_blocks(width, 0, enc_depth - 1, wide=wide, attention=dec_att)
        dec_out = conv2D(width, C, kernel=3, stride=1, bias=False)
        self.decoder = nn.Sequential(*decoder, dec_out, nn.Tanh())

        for i in range(2):
            att = aux_att
            if att is not None:
                att = [a + enc_depth - 1 for a in aux_att]
            aux_branch = stack_enc_blocks(
                width // 2, enc_depth - 1, enc_depth - 1 + aux_depth,
                wide=wide, attention=att)
            if p_emb_drop > 0:
                aux_branch = [nn.Dropout2d(p=p_emb_drop)] + aux_branch
            setattr(self, f'aux_{i}', nn.Sequential(*aux_branch))

        self.pool = MaxMean2D()
        aux_dim = width * 2 ** (enc_depth + aux_depth)
        self.rnn = RNNBlock(aux_dim, rnn_dim, bidirectional=True)

        out_layers = [nn.Linear(self.rnn.out_ch, 1, bias=False)]
        if p_out_drop > 0:
            out_layers = [nn.Dropout(p=p_out_drop)] + out_layers
        self.out = nn.Sequential(*out_layers)
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
            a0 = self.pool(self.aux_0(h0))
            a1 = self.pool(self.aux_1(h1))

            for val, arr in zip([h, a0, a1], [hidden, aux_0, aux_1]):
                val = val.unsqueeze(2)
                arr.append(val)

        hidden = torch.cat(hidden, dim=2)
        x_rec = torch.cat(x_rec, dim=2) if self.is_train else None
        aux_0 = torch.cat(aux_0, dim=2)
        aux_1 = torch.cat(aux_1, dim=2)

        aux_out = torch.cat([aux_0, aux_1], dim=1)
        # N, C, D -> D, N, C
        seq = self.rnn(aux_out)
        y_hat = self.out(seq)
        return hidden, x_rec, y_hat

    @staticmethod
    def to_y(enc: Tensor, x_rec: Tensor, y_hat: Tensor):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred
