import math
import torch

from torch import nn, Tensor
from typing import List, Optional, Tuple

from ..efficient_attention.efficient_attention import EfficientAttention
from ..layers import conv2D, relu, ActivationFn, EncoderBlock, DecoderBlock
from ..ops import select, decide


def stack_enc_blocks(width: int, n: int, wide=False,
                     act_fn: Optional[ActivationFn] = relu,
                     attention: Optional[List[int]] = None):
    if attention is None:
        attention = []
    layers = []
    for i in range(n):
        in_ch = width * 2**i
        out_ch = width * 2**(i+1)
        if i in attention:
            assert not in_ch % 4
            att = EfficientAttention(in_ch, in_ch, in_ch//4, in_ch)
            layers.append(att)
        h_ch = out_ch if wide else in_ch
        block = EncoderBlock(in_ch, out_ch, h_ch, act_fn=act_fn)
        layers.append(block)
    return layers


def stack_dec_blocks(width: int, n: int, wide=False,
                     act_fn: Optional[ActivationFn] = relu,
                     attention: Optional[List[int]] = None):
    layers = []
    for i in sorted(range(n), reverse=True):
        in_ch = width * 2**(i+1)
        out_ch = width * 2**i
        h_ch = in_ch if wide else out_ch
        block = DecoderBlock(in_ch, out_ch, h_ch, act_fn=act_fn)
        layers.append(block)
        if i in attention:
            assert not out_ch % 4
            att = EfficientAttention(out_ch, out_ch, out_ch//4, out_ch)
            layers.append(att)
    return layers


def act_fn_from_str(name: str, neg_slope=0.0):
    if name == 'relu':
        func = nn.ReLU(inplace=False)
    elif name == 'relu_inplace':
        func = nn.ReLU(inplace=True)
    elif name == 'prelu':
        func = nn.PReLU(init=neg_slope)
    elif name == 'lrelu':
        func = nn.LeakyReLU(negative_slope=neg_slope)
    else:
        raise NotImplementedError()
    return func


class Encoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], width: int,
                 depth: int, wide=False,
                 enc_att: Optional[List[int]] = None,
                 dec_att: Optional[List[int]] = None,
                 act_fn: Optional[str] = 'relu',
                 neg_slope: Optional[float] = 0.0, train=True):
        super(Encoder, self).__init__()
        C, H, W = image_shape
        if H != W:
            raise AttributeError("Only square images are supported!")

        max_depth = math.log2(H)
        if depth > max_depth:
            raise AttributeError(
                f"enc_depth should be <= {int(max_depth)} given the "
                f"image_size ({H}, {H})")

        if width % 2:
            raise AttributeError("width must be even number")

        func = act_fn_from_str(act_fn, neg_slope)

        stem = [conv2D(C, width, bias=False), func]
        encoder = stack_enc_blocks(width, depth-1, wide=wide,
                                   act_fn=func, attention=enc_att)
        self.encoder = nn.Sequential(*stem, *encoder)

        decoder = stack_dec_blocks(width, depth-1, wide=wide,
                                   act_fn=func, attention=dec_att)
        dec_out = conv2D(width, C, kernel=3, stride=1, bias=False)
        self.decoder = nn.Sequential(*decoder, dec_out, nn.Tanh())

        self.is_train = train

    def forward(self, x: Tensor, y: Optional[Tensor] = None):
        N, C, D, H, W = x.shape
        hid, x_rec = [], []

        for f in range(D):
            h = self.encoder(x[:, :, f])

            if self.is_train:
                hc = select(h, y)
                x1 = self.decoder(hc).unsqueeze(2)
                x_rec.append(x1)

            h = h.unsqueeze(2)
            hid.append(h)

        hid = torch.cat(hid, dim=2)
        x_rec = torch.cat(x_rec, dim=2) if self.is_train else None
        return hid, x_rec

    @staticmethod
    def to_y(hid: Tensor, x_rec: Tensor):
        y_pred = decide(hid.detach())
        return y_pred.clamp_(0.05, 0.95)
