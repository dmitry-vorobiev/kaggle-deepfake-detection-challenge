import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def identity(x: Tensor) -> Tensor:
    return x


def zeros(n: int, device: torch.device) -> Tensor:
    return torch.zeros(n, dtype=torch.int64, device=device)


def ones(n: int, device: torch.device) -> Tensor:
    return torch.ones(n, dtype=torch.int64, device=device)


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
    return a.clamp_max_(1)


def act_frames(h: Tensor, y: Tensor) -> Tensor:
    N, C, D, H, W = h.shape
    y = reshape_as(y, h)
    h0, h1 = h.chunk(2, dim=1)
    a = h0 * (1 - y) + h1 * y
    n_el = a.numel() / max(N * D, 1)
    a = a.abs().sum((1, 3, 4)) / n_el

    # For simplicity, and without losing generality,
    # we constrain a(x) to be equal to 1
    return a.clamp_max_(1)


def decide(h: Tensor) -> Tensor:
    N = h.size(0)
    with torch.no_grad():
        a0 = act_frames(h, zeros(N, h.device)).unsqueeze(2)
        a1 = act_frames(h, ones(N, h.device)).unsqueeze(2)
        a = torch.cat([a0, a1], dim=2)
        _, y_pred = torch.max(a, dim=2)
        return y_pred.float().mean(1)
