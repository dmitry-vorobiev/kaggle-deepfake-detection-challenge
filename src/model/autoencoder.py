
import torch.nn.functional as F
from functools import partial
from torch import nn, Tensor
from typing import Tuple

from .ops import select


def enc_block(in_ch: int, out_ch: int, kernel=3, stride=2, pad=0, 
              bn=True) -> nn.Module:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, 
                     padding=pad)
    relu = nn.ReLU(inplace=True)
    if bn:
        layers = [conv, nn.BatchNorm2d(out_ch), relu]
    else:
        layers = [conv, relu]
    return nn.Sequential(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel=3, 
                 scale=2, pad=0, bn=True):
        super(DecoderBlock, self).__init__()
        self.upsample = partial(F.interpolate, 
                                scale_factor=scale, 
                                mode='nearest')
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, 
                         stride=1, padding=pad)
        relu = nn.ReLU(inplace=True)
        if bn:
            layers = [conv, nn.BatchNorm2d(out_ch), relu]
        else:
            layers = [conv, relu]
        self.layers = nn.Sequential(*layers)
  
    def forward(self, x) -> Tensor:
        x = self.upsample(x)
        out = self.layers(x)
        return out


class Autoencoder(nn.Module):
    def __init__(self, in_ch: int, depth: int, size=8, pad=1):
        super(Autoencoder, self).__init__()
        self.encoder = Autoencoder._build_encoder(in_ch, depth, size, pad)
        self.decoder = Autoencoder._build_decoder(in_ch, depth, size, pad)
        
    @staticmethod
    def _build_encoder(in_ch: int, depth: int, size: int, pad: int) -> nn.Module:        
        stem = enc_block(in_ch, size, stride=1, pad=pad, bn=False)
        main = [enc_block(size * 2**i, size * 2**(i+1), pad=pad) 
                for i in range(0, depth - 1)]
        return nn.Sequential(stem, *main)
    
    @staticmethod
    def _build_decoder(out_ch: int, depth: int, size: int, pad: int) -> nn.Module:
        main = [DecoderBlock(size * 2**(i+1), size * 2**i, pad=pad) 
                for i in sorted(range(0, depth - 1), reverse=True)]
        last = nn.Conv2d(size, out_ch, 3, stride=1, padding=pad)
        return nn.Sequential(*main, last, nn.Tanh())
        
    def forward(self, x, y) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        hc = select(h, y)
        x_hat = self.decoder(hc)
        return h, x_hat
