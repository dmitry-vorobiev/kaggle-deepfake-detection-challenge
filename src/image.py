import cv2
import h5py
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
def crop_square(img: np.ndarray, bbox: np.ndarray, pad_pct=0.05) -> np.ndarray:
    img_h, img_w, _ = img.shape
    if pad_pct > 0:
        bbox = expand_bbox(bbox, pad_pct)
    x0, y0, x1, y1 = bbox.astype(np.int16)
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


def write_hdf5(path: str, images: List[np.ndarray], append=False,
               opts=dict(compression=6, shuffle=True)) -> None:
    mode = 'a' if append else 'w'
    with h5py.File(path, mode) as file:
        offset = len(file) if append else 0
        for i, image in enumerate(images):
            dataset = file.create_dataset(
                '%03d' % (i + offset), 
                data=cv2.imencode('.png', image)[1], **opts)


def read_hdf5(path: str, num_frames: int) -> List[np.ndarray]:
    images = []
    with h5py.File(path, 'r+') as file:
        total_frames = len(file)
        pick = create_mask(num_frames, total_frames)
        for i, key in enumerate(file.keys()):
            if pick[i]:
                img = np.uint8(file[key])
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images


def calc_idxs(n: int, total: int) -> np.ndarray:
    idxs = np.linspace(0, total, n, dtype=int, endpoint=False)
    rnd_shift = np.random.randint(0, (total - idxs[-1]))
    return idxs + rnd_shift


def create_mask(n: int, total: int) -> np.ndarray:
    mask = np.zeros(total, dtype=np.bool)
    idxs = calc_idxs(n, total)
    mask[idxs] = 1
    return mask

