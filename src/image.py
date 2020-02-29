import numpy as np

from numba import njit
from typing import List, Tuple


@njit
def calc_axis(c0: int, c1: int, pad: int, cmax: int) -> Tuple[int, int, int]:
    c0 = max(0, c0 - pad)
    c1 = min(cmax, c1 + pad)
    return c0, c1, c1 - c0


@njit
def expand_bbox(bbox: np.ndarray, pct: int) -> np.ndarray:
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
