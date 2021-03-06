import os
from typing import List, Optional

import cv2
import h5py
import numpy as np
import pandas as pd

LOSSLESS_OPTIONS = {
    'png': [cv2.IMWRITE_PNG_COMPRESSION, 5],
    'webp': None
}

LOSSY_OPTIONS = {
    'webp': [cv2.IMWRITE_WEBP_QUALITY, 100],
    'jpeg': [cv2.IMWRITE_JPEG_QUALITY, 100],
}


def mkdirs(base_dir: str, chunk_dirs: List[str]) -> None:
    for chunk_dir in chunk_dirs:
        dir_path = os.path.join(base_dir, chunk_dir)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def dump_to_disk(images: List[np.ndarray], dir_path: str, 
                 filename: str, img_format: str, scale=1.,
                 pack=False, lossy=False) -> None:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if len(images) > 0:
        if pack:
            file_path = os.path.join(dir_path, filename+'.h5')
            write_hdf5(file_path, images, img_format, scale, lossy)
        else:
            path = os.path.join(dir_path, filename)
            write_images(path, images, img_format, scale, lossy)
    else:
        raise RuntimeError(
            'No images to write for {}/{}'.format(dir_path, filename))


def resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale != 1.0:
        H, W, C = image.shape
        H = int(H * scale)
        W = int(W * scale)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    return image


def write_images(dir_path: str, images: List[np.ndarray], 
                 img_format: str, scale: float, lossy: bool) -> None:
    if lossy:
        img_opts = LOSSY_OPTIONS[img_format]
    else:
        img_opts = LOSSLESS_OPTIONS[img_format]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for i, image in enumerate(images):
        image = resize(image, scale)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(dir_path, '%03d.%s' % (i, img_format))
        cv2.imwrite(path, image, img_opts)   


def write_hdf5(path: str, images: List[np.ndarray], img_format: str,
               scale: float, lossy: bool) -> None:
    img_ext = '.' + img_format
    if lossy:
        img_opts = LOSSY_OPTIONS[img_format]
    else:
        img_opts = LOSSLESS_OPTIONS[img_format]
    with h5py.File(path, 'w') as file:
        for i, image in enumerate(images):
            image = resize(image, scale)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_bytes = cv2.imencode(img_ext, image, img_opts)[1]
            dataset = file.create_dataset(
                '%03d' % i, data=img_bytes, compression=None, shuffle=False)


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


def get_file_list(df: pd.DataFrame, start: int, end: int,
                  base_dir: str) -> List[str]:
    path_fn = lambda row: os.path.join(base_dir, row.dir, row.name)
    return df.iloc[start:end].apply(path_fn, axis=1).values.tolist()
