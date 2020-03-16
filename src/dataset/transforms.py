import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from functools import partial
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


def simple_transforms(img_size: int, mean: Mean = None, std: Std = None,
                      device: Optional[torch.device] = None,
                      hpf_n=-1) -> Callable[[Any], Tensor]:
    resize = T.Lambda(partial(cv2.resize,
                              dsize=(img_size, img_size),
                              interpolation=cv2.INTER_AREA))
    ops = [resize, T.ToTensor()]
    if device is not None:
        ops.append(partial(Tensor.to, device))
    if hpf_n > 0:
        ops.append(T.Lambda(partial(image_grad, n=hpf_n, keep_size=True)))
    if mean and std:
        if mean is None or std is None:
            raise AttributeError('Please specify both mean and std')
        ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)
