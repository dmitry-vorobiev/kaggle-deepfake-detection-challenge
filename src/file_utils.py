import os
from typing import List

import cv2
import h5py
import numpy as np


def mkdirs(base_dir: str, chunk_dirs: List[str]) -> None:
    for chunk_dir in chunk_dirs:
        dir_path = os.path.join(base_dir, chunk_dir)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def dump_to_disk(images: List[np.ndarray], 
                 dir_path: str, filename: str, append=False) -> None:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, filename+'.h5')
    write_hdf5(file_path, images, append=append)


def write_hdf5(path: str, images: List[np.ndarray], append=False,
               opts=dict(compression=6, shuffle=True)) -> None:
    mode = 'a' if append else 'w'
    with h5py.File(path, mode) as file:
        offset = len(file) if append else 0
        for i, image in enumerate(images):
            dataset = file.create_dataset(
                '%03d' % (i + offset), 
                data=cv2.imencode('.png', image)[1], **opts)


def read_hdf5(path: str, num_frames=30) -> List[np.ndarray]:
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
