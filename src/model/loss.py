import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor
from typing import Dict, Tuple

from . import ModelOut
from .ops import act, ones, zeros


def activation_loss(x: Tensor, y: LongTensor) -> Tensor:
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
    
    loss1 = activation_loss(h, y)
    loss2 = F.l1_loss(x_hat, x, reduction='mean') * 0.1
    loss3 = F.binary_cross_entropy_with_logits(y_hat.squeeze(1), y.float())
    
    return loss1 + loss2 + loss3


class ForensicTransferLoss:
    def __init__(self, act_w: int, rec_w: int):
        self.act_w = act_w
        self.rec_w = rec_w

    def __call__(self, out: Tuple[FloatTensor, FloatTensor],
                 x: FloatTensor, y: LongTensor) -> Dict[str, Tensor]:
        h, x_hat = out
        act_loss = activation_loss(h, y)
        rec_loss = F.l1_loss(x_hat, x, reduction='mean')
        total_loss = act_loss * self.act_w + rec_loss * self.rec_w
        out = dict(
            loss=total_loss,
            act_loss=act_loss,
            rec_loss=rec_loss)
        return out


class TripleLoss(ForensicTransferLoss):
    def __init__(self, act_w: int, rec_w: int, bce_w: int):
        super(TripleLoss, self).__init__(act_w, rec_w)
        self.bce_w = bce_w

    def __call__(self, out: ModelOut, x: FloatTensor, y: LongTensor) -> Dict[str, Tensor]:
        h, x_hat, y_hat = out
        out = super().__call__((h, x_hat), x, y)
        bce_loss = F.binary_cross_entropy_with_logits(
            y_hat.squeeze(1), y.float())
        out['loss'] += bce_loss * self.bce_w
        out['bce_loss'] = bce_loss
        return out
