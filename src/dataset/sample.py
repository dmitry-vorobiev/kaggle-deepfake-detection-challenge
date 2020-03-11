import numpy as np
import torch
from functools import partial
from typing import Callable, Tuple


def sparse_frames(n: int, total: int) -> np.ndarray:
    idxs = np.linspace(0, total, min(n, total), dtype=int, endpoint=False)
    rnd_shift = np.random.randint(0, (total - idxs[-1]))
    return idxs + rnd_shift


def rnd_slice_frames(n: int, total: int, stride=1.) -> np.ndarray:
    idxs = np.arange(0, total, stride)[:n].astype(np.uint16)
    rnd_shift = np.random.randint(0, (total - idxs[-1]))
    return idxs + rnd_shift


class FrameSampler():
    def __init__(self, num_frames: int, real_fake_ratio: float, 
                 p_sparse: float):
        self.num_frames = num_frames
        self.real_fake_ratio = real_fake_ratio
        self.p_sparse = p_sparse
        
    def __call__(self, label: Tuple[int, bool]) -> Callable[[int], np.ndarray]:
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


class BalancedSampler(torch.utils.data.RandomSampler):
    def __init__(self, data_source: torch.utils.data.Dataset, 
                 replacement=False, num_samples=None):
        super().__init__(data_source, replacement, num_samples)
        if not hasattr(data_source, 'df'):
            raise ValueError("DataSource must have a 'df' property")
            
        if not 'label' in data_source.df: 
            raise ValueError("DataSource.df must have a 'label' column")
    
    def __iter__(self):
        df = self.data_source.df
        all_labels = df['label'].values
        uniq_labels, label_freq = np.unique(all_labels, return_counts=True)
        rev_freq = (len(all_labels) / label_freq)
        shuffle = np.random.permutation
        
        idxs = []
        for freq, label in zip(rev_freq, uniq_labels):
            fraction, times = np.modf(freq)
            label_idxs = (all_labels == label).nonzero()[0]
            for _ in range(int(times)):
                label_idxs = shuffle(label_idxs)
                idxs.append(label_idxs)
            if fraction > 0.05:
                label_idxs = shuffle(label_idxs)
                chunk = int(len(label_idxs) * fraction)
                idxs.append(label_idxs[:chunk])
        idxs = np.concatenate(idxs)
        idxs = shuffle(idxs)[:self.num_samples]
#         for i in idxs:
#             yield i 
        return iter(idxs.tolist())
