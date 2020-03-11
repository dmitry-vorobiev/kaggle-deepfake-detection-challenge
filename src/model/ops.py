import torch
from torch import Tensor


def identity(x: Tensor) -> Tensor:
    return x


def pool_gru(out_gru: Tuple[Tensor, Tensor]) -> Tensor:
    out, _ = out_gru
    out_avg = torch.mean(out, dim=1)
    out_max, _ = torch.max(out, dim=1)
    return torch.cat([out_avg, out_max], dim=1)


def pool_lstm(out_lstm: Tuple[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
    out, (out_h, out_c) = out_lstm
    out_avg = torch.mean(out, 1)
    out_max, _ = torch.max(out, 1)
    out_h = out_h.squeeze(0)
    return torch.cat([out_avg, out_max, out_h], dim=1)


def select(h: Tensor, y: Tensor) -> Tensor:
    N, C, H, W = h.shape
    y = y.reshape(N, 1, 1, 1)
    
    h0, h1 = h.chunk(2, dim=1)
    h0 = h0 * (1 - y)
    h1 = h1 * y
    h = torch.cat([h0, h1], dim=1)
    return h


def act(h: Tensor, y: Tensor) -> Tensor:
    N = y.size(0)
    dims = [1] * (h.ndim-1)
    y = y.reshape(N, *dims)
    
    h0, h1 = h.chunk(2, dim=-3)
    a = h0 * (1 - y) + h1 * y
    n_el = a.numel() / max(N, 1)
    a = a.abs().sum(tuple(range(1, a.ndim))) / n_el
    
    # For simplicity, and without losing generality, 
    # we constrain a(x) to be equal to 1
    return a.clamp_max_(1).ceil_()
