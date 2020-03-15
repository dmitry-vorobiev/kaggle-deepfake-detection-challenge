import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


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


def reshape_as(y: Tensor, h: Tensor) -> Tensor:
    N = y.size(0)
    dims = [1] * (h.ndim-1)
    y = y.reshape(N, *dims)
    return y


def select(h: Tensor, y: Tensor) -> Tensor:
    C = h.size(1)
    C_mid = C // 2
    y = reshape_as(y, h)
    h = h.clone()
    h[:, 0:C_mid] *= (1 - y)
    h[:, C_mid:C] *= y
    return h


def act(h: Tensor, y: Tensor) -> Tensor:
    N = y.size(0)
    y = reshape_as(y, h)
    h0, h1 = h.chunk(2, dim=1)
    a = h0 * (1 - y) + h1 * y
    n_el = a.numel() / max(N, 1)
    a = a.abs().sum(tuple(range(1, a.ndim))) / n_el
    
    # For simplicity, and without losing generality, 
    # we constrain a(x) to be equal to 1
    return a.clamp_max_(1).ceil_()


def diff(x: Tensor, dim: int) -> Tensor:
    mask = list(map(slice, x.shape[:dim]))
    mask0 = mask + [slice(1, x.size(dim))]
    mask1 = mask + [slice(0, -1)]
    return x[mask0] - x[mask1]


def image_grad(x: Tensor, n=1, keep_size=False) -> Tensor:
    for _ in range(n):
        x = diff(x, -1)
        x = diff(x, -2)
    if keep_size:
        pad = [(n + i) // 2 for i in [0, 1, 0, 1]]
        x = F.pad(x, pad)
    return x
