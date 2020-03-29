import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple

Mean = Optional[Tuple[float, float, float]]
Std = Mean


def no_transforms(image: np.ndarray, size=256) -> Tensor:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return to_tensor(image)


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


class Resize(object):
    def __init__(self, size: int, mode=cv2.INTER_NEAREST):
        if not size:
            raise AttributeError("Size should be positive number")
        self.size = size
        self.mode = mode

    def __call__(self, image: np.ndarray):
        size = (self.size, self.size)
        return cv2.resize(image, dsize=size, interpolation=self.mode)

    def __repr__(self):
        return "{}(size={})".format(Resize.__name__, self.size)


class ResizeTensor(object):
    def __init__(self, size: int, mode='nearest', normalize=True):
        if not size:
            raise AttributeError("Size should be positive number")
        self.size = size
        self.mode = mode
        self.normalize = normalize
        self.align_corners = False
        if mode in ['bilinear', 'bicubic']:
            # not sure if this is needed
            self.align_corners = True

    def __call__(self, t: Tensor):
        t = t.float().unsqueeze_(0)
        t = F.interpolate(t, size=self.size, mode=self.mode, align_corners=self.align_corners)
        t = t.clamp_(min=0, max=255).squeeze_(0)
        if self.normalize:
            return t.div_(255)
        return t.round_().byte()

    def __repr__(self):
        return "{}(size={}, mode={})".format(Resize.__name__, self.size, self.mode)


class PadIfNeeded(object):
    def __init__(self, size: int, mode='constant', value=0, normalize=True):
        if not size:
            raise AttributeError("Size should be positive number")
        self.size = size
        self.mode = mode
        self.value = value
        self.normalize = normalize

    def __call__(self, t: Tensor):
        S = self.size
        H, W = t.size(-2), t.size(-1)
        if H >= S and W >= S:
            return t
        pad_H = (S - H) // 2
        pad_W = (S - W) // 2
        pad = [pad_W, S - (W + pad_W),
               pad_H, S - (H + pad_H)]
        if self.mode != 'constant':
            t = t.float().unsqueeze_(0)
        t = F.pad(t, pad, mode=self.mode, value=self.value)
        if t.ndim > 3:
            t = t.squeeze_(0)
        if self.normalize:
            return t.float().div_(255)
        return t

    def __repr__(self):
        return "{}(size={}, mode={}, value={})".format(
            PadIfNeeded.__name__, self.size, self.mode, self.value)


class CropCenter(object):
    def __init__(self, size: int):
        if not size:
            raise AttributeError("Size should be positive number")
        self.size = size

    def __call__(self, t: Tensor):
        S = int(self.size)
        H, W = t.size(-2), t.size(-1)
        y0 = int(round((H - S) / 2.))
        x0 = int(round((W - S) / 2.))
        t = t[:, y0:y0+S, x0:x0+S]
        return t

    def __repr__(self):
        return "{}(size={})".format(CropCenter.__name__, self.size)


class SpatialGradFilter(object):
    def __init__(self, order: int):
        if not order:
            raise AttributeError("Order should be positive number")
        self.order = order

    def __call__(self, t: Tensor):
        return image_grad(t, n=self.order, keep_size=True)

    def __repr__(self):
        return "{}(order={})".format(SpatialGradFilter.__name__, self.order)


class RandomHorizontalFlipSequence(object):
    def __init__(self, p=0.5):
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p

    def __call__(self, t: Tensor):
        if random.uniform(0, 1) < self.p:
            return t.flip(-1)
        return t

    def __repr__(self):
        return "{}(p={})".format(RandomHorizontalFlipSequence.__name__, self.p)
