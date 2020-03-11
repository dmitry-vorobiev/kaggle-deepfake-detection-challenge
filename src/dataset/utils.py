import numpy as np
import os
import pandas as pd
from typing import List, Optional, Union


def read_labels(base_path: str, chunk_dirs: Optional[List[str]]=None, 
                label: Optional[int]=None) -> pd.DataFrame:
    if not os.path.isdir(base_path):
        raise ValueError('Invalid data dir')
    if not chunk_dirs:
        chunk_dirs = os.listdir(base_path)
    labels = []
    for dir_name in chunk_dirs:
        path = os.path.join(base_path, dir_name, 'metadata.json')
        df = pd.read_json(path).T
        df['dir'] = dir_name
        df['label'] = (df['label'] == 'FAKE').astype(np.uint8)
        df.drop(['split'], axis=1, inplace=True)
        if label is not None:
            mask = df['label'] == label
            df = df[mask]
        labels.append(df)
    return pd.concat(labels)


def create_mask(idxs: np.ndarray, total: int) -> np.ndarray:
    mask = np.zeros(total, dtype=np.bool)
    mask[idxs] = 1
    return mask


def pad(frames: np.ndarray, amount: int, where :str='start') -> np.ndarray:
    dims = np.zeros((frames.ndim, 2), dtype=np.int8)
    pad_dim = 1 if where == 'end' else 0
    dims[0, pad_dim] = amount
    return np.pad(frames, dims, 'constant')
