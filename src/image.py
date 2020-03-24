import numpy as np

from numba import njit
import torch
from torch import IntTensor, FloatTensor, Tensor
from typing import List, Tuple


@njit
def calc_axis(c0: int, c1: int, pad: int, cmax: int) -> Tuple[int, int, int]:
    c0 = max(0, c0 - pad)
    c1 = min(cmax, c1 + pad)
    return c0, c1, c1 - c0


@njit
def expand_bbox(bbox: np.ndarray, pct: float) -> np.ndarray:
    bbox = np.copy(bbox)
    bbox[:2] *= 1 - pct
    bbox[2:] *= 1 + pct
    return bbox


@njit
def fix_coords(bbox: np.ndarray, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox.astype(np.int16)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(x1, img_width), min(y1, img_height)
    return x0, y0, x1, y1


@njit
def square_bbox(img_h, img_w, bbox: np.ndarray, pad_pct=0.05) -> np.ndarray:
    pass


@njit
def crop_square(img: np.ndarray, bbox: np.ndarray, pad_pct=0.05) -> np.ndarray:
    img_h, img_w, _ = img.shape
    if pad_pct > 0:
        bbox = expand_bbox(bbox, pad_pct)
    x0, y0, x1, y1 = fix_coords(bbox, img_w, img_h)
    w, h = x1 - x0, y1 - y0
    if w > h:
        pad = (w - h) // 2
        y0, y1, h = calc_axis(y0, y1, pad, img_h)
    elif h > w:
        pad = (h - w) // 2
        x0, x1, w = calc_axis(x0, x1, pad, img_w)
    size = min(w, h)
    face = img[y0:y1, x0:x1][:size, :size]
    return face


def calc_axis_torch(c0: Tensor, c1: Tensor, pad: Tensor,
                    cmax: int) -> Tuple[Tensor, ...]:
    c0 = max(0, c0 - pad)
    c1 = min(cmax, c1 + pad)
    return c0, c1, c1 - c0


def expand_bbox_torch(bbox: Tensor, pct: float) -> Tensor:
    bbox = bbox.clone().detach()
    bbox[:2] *= 1 - pct
    bbox[2:] *= 1 + pct
    return bbox


def fix_coords_torch(bbox: Tensor, img_width: int,
                     img_height: int) -> Tuple[Tensor, ...]:
    x0, y0, x1, y1 = bbox.int()
    x0.clamp_min_(0)
    y0.clamp_min_(0)
    x1.clamp_max_(img_width)
    y1.clamp_max_(img_height)
    return x0, y0, x1, y1


def crop_square_torch(img: Tensor, bbox: Tensor, pad_pct=0.05) -> Tensor:
    C, H, W = img.shape
    if pad_pct > 0:
        bbox = expand_bbox_torch(bbox, pad_pct)
    x0, y0, x1, y1 = fix_coords_torch(bbox, W, H)
    w, h = x1 - x0, y1 - y0
    if w > h:
        pad = (w - h) // 2
        y0, y1, h = calc_axis_torch(y0, y1, pad, H)
    elif h > w:
        pad = (h - w) // 2
        x0, x1, w = calc_axis_torch(x0, x1, pad, W)
    size = min(w, h)
    face = img[:, y0:y1, x0:x1][:, :size, :size]
    return face


# @torch.jit.script
# def calc_axis_torch(c0: Tensor, c1: Tensor, pad: int,
#                     cmax: int):
#     c0 = torch.max(torch.zeros_like(c0), c0 - pad)
#     c1 = torch.min(torch.full_like(c1, cmax), c1 + pad)
#     return c0, c1, c1 - c0
#
#
# @torch.jit.script
# def expand_bbox_torch(bbox: Tensor, pct: float) -> Tensor:
#     bbox = bbox.clone().detach()
#     bbox[:2] *= 1 - pct
#     bbox[2:] *= 1 + pct
#     return bbox
#
#
# @torch.jit.script
# def fix_coords_torch(bbox: Tensor, img_width: int, img_height: int) -> Tensor:
#     bbox = bbox.int()
#     bbox[0].clamp_min_(0)
#     bbox[1].clamp_min_(0)
#     bbox[2].clamp_max_(img_width)
#     bbox[3].clamp_max_(img_height)
#     return bbox
#
#
# @torch.jit.script
# def crop_square_torch(img: Tensor, bbox: Tensor, pad_pct=torch.tensor(0.05)) -> Tensor:
#     C, H, W = img.shape
#     if pad_pct > 0:
#         bbox = expand_bbox_torch(bbox, pad_pct)
#     bbox = fix_coords_torch(bbox, W, H)
#     x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
#     w, h = x1 - x0, y1 - y0
#     if w > h:
#         pad = (w - h) // 2
#         y0, y1, h = calc_axis_torch(y0, y1, pad, H)
#     elif h > w:
#         pad = (h - w) // 2
#         x0, x1, w = calc_axis_torch(x0, x1, pad, W)
#     size = min(w, h)
#     face = img[:, y0:y1, x0:x1][:, :size, :size]
#     return face
