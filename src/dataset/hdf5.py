import cv2
import h5py
import numpy as np
import os
import pandas as pd
import torch
from torch import FloatTensor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .sample import FrameSampler
from .utils import create_mask, pad_torch


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path: str, size: Tuple[int, int], 
                 sampler: FrameSampler, 
                 x_tfms: Callable[[np.ndarray], FloatTensor],
                 sub_dirs: Optional[List[str]]=None):
        super(HDF5Dataset, self).__init__()    
        self.base_path = base_path
        self.size = size
        self.sampler = sampler
        self.x_tfms = x_tfms
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
    def read_hdf5(path: str, num_frames: int, size: int,
                  sample_fn: Callable[[int], np.ndarray]) -> np.ndarray:
        img_size = (size, size)
        images = []
        with h5py.File(path, 'r') as file:
            total_frames = len(file)
            if total_frames > 0:
                idxs = sample_fn(total_frames)
                pick = create_mask(idxs, total_frames)
                keys = iter(file.keys())
                for i in range(idxs[-1] + 1):
                    key = next(keys)
                    if pick[i]:
                        img = HDF5Dataset._proc_image(file[key], img_size)
                        images.append(img)
                return np.stack(images)
            else:
                return np.empty((0, size, size, 3), dtype=np.uint8)

    @staticmethod
    def read_hdf5_alt(path: str, num_frames: int, size: int,
                  sample_fn: Callable[[int], np.ndarray]) -> np.ndarray:
        img_size = (size, size)
        images = []
        with h5py.File(path, 'r') as file:
            total_frames = len(file)
            if total_frames > 0:
                idxs = sample_fn(total_frames)
                for i in idxs:
                    key = '%03d' % i
                    img = HDF5Dataset._proc_image(file[key], img_size)
                    images.append(img)
        return images
    
    @staticmethod
    def _proc_image(img: h5py.Dataset, 
                    img_size: Tuple[int, int]) -> np.ndarray:
        img = np.uint8(img)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size, 
                         interpolation=cv2.INTER_NEAREST)
        return img
        
    def __len__(self) :
        return len(self.df)
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        num_frames, size = self.size
        meta = self.df.iloc[idx]
        label = int(meta.label)
        path = os.path.join(self.base_path, meta.dir, meta.video)
        
        if os.path.isfile(path):
            sample_fn = self.sampler(meta.label)
            frames = HDF5Dataset.read_hdf5_alt(
                path, num_frames, size, sample_fn=sample_fn)
        else:
            print('Unable to read {}'.format(path))
            frames = []

        if len(frames) > 0:
            if self.x_tfms:
                frames = [self.x_tfms(frame) for frame in frames]
            frames = torch.stack(frames)
            pad_amount = num_frames - len(frames)
            if pad_amount > 0:
                frames = pad_torch(frames, pad_amount, 'start')
        else:
            print('Empty file {}'.format(path))
            frames =  torch.zeros((num_frames, 3, size, size), dtype=torch.float32)
        return frames, label
