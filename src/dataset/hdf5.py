import cv2
import h5py
import numpy as np
import os
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .sample import FrameSampler
from .transforms import no_transforms
from .utils import pad_torch

Transforms = Callable[[np.ndarray], Tensor]


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path: str, frames: int, sampler: FrameSampler,
                 transforms: Optional[Transforms] = None,
                 sub_dirs: Optional[List[str]] = None):
        super(HDF5Dataset, self).__init__()
        self.base_path = base_path
        self.frames = frames
        self.sampler = sampler
        self.transforms = transforms
        self.df = HDF5Dataset._read_annotations(base_path, sub_dirs)
        
    @staticmethod
    def _read_annotations(base_path: str, 
                          sub_dirs: Optional[List[str]]) -> pd.DataFrame:
        if not os.path.isdir(base_path):
            raise RuntimeError('Unable to access %s' % base_path)
        parts = []
        load_all = sub_dirs is None
        if load_all:
            sub_dirs = os.listdir(base_path)
        for chunk_dir in sub_dirs:
            chunk_path = Path(base_path)/chunk_dir
            if not chunk_path.is_dir():
                if not load_all:
                    print('Invalid dir: %s' % str(chunk_path))
                continue
            files = os.listdir(chunk_path)
            df = pd.DataFrame(files, columns=['video'])
            df['label'] = df['video'].str.endswith('_1.h5')
            df['dir'] = chunk_dir
            parts.append(df)
        if len(parts) < 1:
            raise AttributeError('No files were found')
        return pd.concat(parts).reset_index()

    @staticmethod
    def read_hdf5(path: str, sample_fn: Callable[[int], np.ndarray]) -> List[np.ndarray]:
        images = []
        with h5py.File(path, 'r') as file:
            total_frames = len(file)
            if total_frames > 0:
                indices = sample_fn(total_frames)
                for i in indices:
                    key = '%03d' % i
                    img = HDF5Dataset._read_image(file[key])
                    images.append(img)
        return images

    @staticmethod
    def _read_image(img: h5py.Dataset) -> np.ndarray:
        img = np.uint8(img)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        num_frames = self.frames
        meta = self.df.iloc[idx]
        label = int(meta.label)
        path = os.path.join(self.base_path, meta.dir, meta.video)
        
        if os.path.isfile(path):
            sample_fn = self.sampler(meta.label)
            frames = HDF5Dataset.read_hdf5(path, sample_fn)
        else:
            print('Unable to read {}'.format(path))
            frames = []

        if len(frames) > 0:
            transform_x = self.transforms or no_transforms
            frames = torch.stack(list(map(transform_x, frames)))
            pad_amount = num_frames - frames.size(0)
            if pad_amount > 0:
                frames = pad_torch(frames, pad_amount, 'start')
            # D, C, H, W -> C, D, H, W
            frames = frames.permute(1, 0, 2, 3)
        else:
            print('Unable to load images from {}'.format(path))
            new_idx = np.random.randint(0, len(self))
            return self[new_idx]
        return frames, label
