import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, RandomSampler
from typing import Any, Callable, Union


def sparse_frames(n: int, total: int) -> np.ndarray:
    indices = np.linspace(0, total, min(n, total), dtype=int, endpoint=False)
    rnd_shift = np.random.randint(0, (total - indices[-1]))
    return indices + rnd_shift


def rnd_slice_frames(n: int, total: int, stride=1.) -> np.ndarray:
    indices = np.arange(0, total, stride)[:n].astype(np.uint16)
    rnd_shift = np.random.randint(0, (total - indices[-1]))
    return indices + rnd_shift


class FrameSampler:
    def __init__(self, num_frames: int, real_fake_ratio: float, 
                 p_sparse: float):
        self.num_frames = num_frames
        self.real_fake_ratio = real_fake_ratio
        self.p_sparse = p_sparse
        
    def __call__(self, label: Union[int, bool]) -> Callable[[int], np.ndarray]:
        dice = np.random.rand()
        if dice < self.p_sparse:
            return partial(sparse_frames, self.num_frames)
        else:
            # Stored frames: fake - 30, real - 150, 
            # the real_fake_ratio should be set to 150 / 30 = 5
            # stride for fake: 5 - (4 * 1) = 1
            # stride for real: 5 - (4 * 0) = 5
            n = self.real_fake_ratio
            stride = n - ((n-1) * int(label))
            return partial(rnd_slice_frames, self.num_frames, stride=stride)


class BalancedSampler(RandomSampler):
    def __init__(self, data_source: Dataset, replacement=False, num_samples=None):
        super().__init__(data_source, replacement, num_samples)
        if not hasattr(data_source, 'df'):
            raise ValueError("DataSource must have a 'df' property")
            
        if 'label' not in data_source.df:
            raise ValueError("DataSource.df must have a 'label' column")
    
    def __iter__(self):
        df = self.data_source.df
        all_labels = df['label'].values
        unique_labels, label_freq = np.unique(all_labels, return_counts=True)
        rev_freq = (len(all_labels) / label_freq)
        shuffle = np.random.permutation
        
        indices = []
        for freq, label in zip(rev_freq, unique_labels):
            fraction, times = np.modf(freq)
            label_indices = (all_labels == label).nonzero()[0]
            for _ in range(int(times)):
                label_indices = shuffle(label_indices)
                indices.append(label_indices)
            if fraction > 0.05:
                label_indices = shuffle(label_indices)
                chunk = int(len(label_indices) * fraction)
                indices.append(label_indices[:chunk])
        indices = np.concatenate(indices)
        indices = shuffle(indices)[:self.num_samples]
        return iter(indices.tolist())
