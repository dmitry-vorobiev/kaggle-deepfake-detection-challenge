import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor
from typing import Tuple

from . import ModelOut
from .ops import act


def zeros(n: int, device: torch.device) -> Tensor:
    return torch.zeros(n, dtype=torch.int64, device=device)


def ones(n: int, device: torch.device) -> Tensor:
    return torch.ones(n, dtype=torch.int64, device=device)


def act_loss(x: Tensor, y: LongTensor) -> Tensor:
    device = x.device
    pos = y.nonzero().reshape(-1)
    neg = (y - 1).nonzero().reshape(-1)
    x0, x1 = x[neg], x[pos]
    n0, n1 = x0.size(0), x1.size(0)
    
    a0_x0 = act(x0, zeros(n0, device))
    a1_x0 = act(x0, ones(n0, device))
    
    a1_x1 = act(x1, ones(n1, device))
    a0_x1 = act(x1, zeros(n1, device))
    
    neg_loss = (a0_x0 - 1).abs() + a1_x0
    pos_loss = (a1_x1 - 1).abs() + a0_x1

    return (neg_loss.sum() + pos_loss.sum()) / y.size(0)


def combined_loss(out: ModelOut, x: FloatTensor, y: LongTensor) -> Tensor:
    h, x_hat, y_hat = out
    
    loss1 = act_loss(h, y)
    loss2 = F.l1_loss(x_hat, x, reduction='mean') * 0.1
    loss3 = F.binary_cross_entropy_with_logits(y_hat.squeeze(1), y.float())
    
    return loss1 + loss2 + loss3
