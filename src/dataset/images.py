import cv2
import numpy as np
import os
import pandas as pd
import torch
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .sample import FrameSampler
from .utils import create_mask, pad


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, base_path: str, size: Tuple[int, int], 
                 sampler: FrameSampler, 
                 sub_dirs: Optional[List[str]]=None):
        self.base_path = base_path
        self.size = size
        self.sampler = sampler
        self.df = ImagesDataset._read_annotations(base_path, sub_dirs)
        
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
            df['label'] = df['video'].str.endswith('_1')
            df['dir'] = chunk_dir
            parts.append(df)
        if len(parts) < 1:
            raise AttributeError('No images were found')
        return pd.concat(parts).reset_index()
    
    @staticmethod
    def read_image_folder(path: str, num_frames: int, size: int,
                          sample_fn: Callable[[int], np.ndarray]) -> np.ndarray:
        img_size = (size, size)
        images = []
        files = sorted(os.listdir(path))
        total_frames = len(files)
        if total_frames > 0:
            idxs = sample_fn(total_frames)
            pick = create_mask(idxs, total_frames)
            for i, file in enumerate(files):
                if pick[i]:
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size, 
                                     interpolation=cv2.INTER_NEAREST)
                    images.append(img)
            return np.stack(images)
        else:
            return np.empty((0, size, size, 3), dtype=np.uint8)
        
    def __len__(self) :
        return len(self.df)
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        num_frames, size = self.size
        meta = self.df.iloc[idx]
        label = int(meta.label)
        path = os.path.join(self.base_path, meta.dir, meta.video)
        
        if os.path.isdir(path):
            sample_fn = self.sampler(meta.label)
            frames = ImagesDataset.read_image_folder(
                path, num_frames, size, sample_fn=sample_fn)
        else:
            print('Dir not found: {}'.format(path))
            frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
        
        if len(frames) > 0:
            pad_amount = num_frames - len(frames)
            if pad_amount > 0:
                frames = pad(frames, pad_amount, 'start')
        else:
            print('Empty file {}'.format(path))
            frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
        return frames, label
