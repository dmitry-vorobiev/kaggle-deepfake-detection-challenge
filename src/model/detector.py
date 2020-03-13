import math
import torch
from torch import nn, Tensor, FloatTensor, LongTensor
from typing import Callable, Tuple, Union

from .autoencoder import AutoEncoder
from .layers import conv3D
from .ops import identity, pool_gru

DetectorOut = Tuple[FloatTensor, FloatTensor, FloatTensor]


class FakeDetector(nn.Module):
    def __init__(self, img_size: int, enc_dim: Tuple[int, int], 
                 seq_size: Tuple[int, int], pool='conv'):
        super(FakeDetector, self).__init__()
        self.autoenc = FakeDetector._build_encoder(img_size, enc_dim)
        seq_in, seq_out = seq_size
        self.pool = FakeDetector._build_pooling(img_size, enc_dim, seq_in, pool)
        self.seq_model = nn.GRU(seq_in, seq_out)
        self.out = nn.Linear(seq_out*2, 1, bias=False)
        
    @staticmethod
    def _build_encoder(img_size: int, enc_dim: Tuple[int, int]) -> AutoEncoder:
        depth, size = enc_dim
        if img_size % 32:
            raise AttributeError('Image size should be a multiple of 32')  
        return AutoEncoder(in_ch=3, depth=depth, size=size, pad=1)
    
    @staticmethod
    def _validate(n: float):
        if math.modf(n)[0] > 0 or n < 1:
            raise AttributeError(
                'Sequence input size is incompatible with encoder dims')
    
    @staticmethod
    def _build_pooling(img_size: int, enc_dim: Tuple[int, int], 
                       seq_in: int, pool: str
                       ) -> Union[nn.Module, Callable[[Tensor], Tensor]]:
        enc_depth, enc_size = enc_dim
        size_factor = 2**(enc_depth-1)
        if size_factor > img_size:
            raise AttributeError(
                'Encoder dims (%d, %d) are incompatible with image '
                'size (%d, %d)' % (enc_depth, enc_size, img_size, img_size))
        emb_S = img_size // size_factor
        emb_C = enc_size * size_factor
        
        if emb_C * emb_S**2 == seq_in:
            return identity
        elif pool == 'conv':
            n = math.log2((emb_C * emb_S**2) / seq_in) / 2
            FakeDetector._validate(n)
            n = int(n)
            print('Using Conv3D pooling: {} layers'.format(n))
            conv = [conv3D(emb_C, emb_C, stride=2) for _ in range(n)]
            return nn.Sequential(*conv)
        else:
            out_S = math.sqrt(seq_in / emb_S / enc_size)
            FakeDetector._validate(out_S)
            out_S = int(out_S)
            in_dim = (emb_C, emb_S, emb_S)
            out_dim = (emb_C, out_S, out_S)
            print('Using avg pooling: {} -> {}'.format(in_dim, out_dim))
            return nn.AdaptiveAvgPool3d(out_dim)
    
    def forward(self, x: FloatTensor, y: LongTensor) -> DetectorOut:
        N, C, D, H, W = x.shape
        hidden, xs_hat = [], []
        
        for f in range(D):
            h, x_hat = self.autoenc(x[:, :, f], y)
            hidden.append(h[:, :, None])
            xs_hat.append(x_hat[:, :, None])
            
        hidden = torch.cat(hidden, dim=2)
        xs_hat = torch.cat(xs_hat, dim=2)
        
        seq = self.pool(hidden).reshape(N, D, -1)
        seq_out = self.seq_model(seq)
        seq_out = pool_gru(seq_out)
        y_hat = self.out(seq_out)
        
        return hidden, xs_hat, y_hat


def basic_detector_256():
    return FakeDetector(img_size=256, enc_dim=(5, 8), seq_size=(2048, 64))


def wide_detector_256():
    return FakeDetector(img_size=256, enc_dim=(5, 16), seq_size=(4096, 64))


def deep_detector_256():
    return FakeDetector(img_size=256, enc_dim=(9, 8), seq_size=(2048, 64))


def deep_wide_detector_256():
    return FakeDetector(img_size=256, enc_dim=(9, 16), seq_size=(4096, 64))
