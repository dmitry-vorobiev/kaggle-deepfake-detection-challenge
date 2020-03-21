import numpy as np
import torch.distributed as dist

from functools import partial
from torch.utils.data import Sampler, Dataset
from typing import Callable, Union


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


class BalancedSampler(Sampler):
    def __init__(self, data_source: Dataset, num_samples=None,
                 rank=None, num_replicas=None):
        super().__init__(data_source)
        if not hasattr(data_source, 'df'):
            raise ValueError("DataSource must have a 'df' property")
        if 'label' not in data_source.df:
            raise ValueError("DataSource.df must have a 'label' column")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.df = data_source.df
        self.num_samples = num_samples
        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = 0

    def __len__(self):
        return self.num_samples or len(self.df)

    def __iter__(self):
        np.random.seed(self.epoch)
        labels = self.df['label'].values
        unique_labels, label_freq = np.unique(labels, return_counts=True)
        rev_freq = (len(labels) / label_freq)
        shuffle = np.random.permutation

        sampled = []
        for freq, label in zip(rev_freq, unique_labels):
            fraction, times = np.modf(freq)
            idxs = (labels == label).nonzero()[0]

            if self.num_replicas > 1:
                offset = self.rank
                chunk_size = len(idxs) // self.num_replicas
                start = chunk_size * offset
                end = chunk_size * (offset + 1)
                idxs = idxs[start:end]

            for _ in range(int(times)):
                idxs = shuffle(idxs)
                sampled.append(idxs)
            if fraction > 0.05:
                idxs = shuffle(idxs)
                chunk = int(len(idxs) * fraction)
                sampled.append(idxs[:chunk])
        indices = np.concatenate(sampled)
        indices = shuffle(indices)[:len(self)]
        return iter(indices.tolist())

    def set_epoch(self, epoch):
        self.epoch = epoch
