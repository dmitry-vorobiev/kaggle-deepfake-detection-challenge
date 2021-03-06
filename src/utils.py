import os
import torchvision.transforms as T

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import List, Optional, Tuple

from dataset import HDF5Dataset, ImagesDataset, FrameSampler, BalancedSampler
from dataset.hdf5 import Transforms2D, Transforms3D


def read_file_list(conf: DictConfig, title: str) -> List[str]:
    dirs = []
    if 'chunks' in conf.keys():
        chunks = conf.chunks
        if isinstance(chunks, int):
            chunks = [chunks]
        elif isinstance(chunks, str):
            interval = list(map(str.strip, chunks.split('-')))
            if len(interval) == 2:
                interval = map(int, interval)
                chunks = list(range(*interval))
            elif len(interval) < 2:
                chunks = list(map(str.strip, chunks.split(',')))
            else:
                raise AttributeError(
                    "Config: incorrect format for 'data.{}.chunks'".format(title))
        for c in chunks:
            dirs.append('dfdc_train_part_{}'.format(c))
    if 'dir_list' in conf.keys() and len(conf.dir_list):
        with open(conf.dir_list) as h:
            path = h.readline()
            if os.path.isdir(path):
                dirs.append(path)
    return dirs


def create_transforms(conf: DictConfig) -> Tuple[Transforms2D, Optional[Transforms3D]]:
    transforms = T.Compose([instantiate(val['transform']) for val in conf.transforms])
    transforms_3d = None
    if 'transforms_3d' in conf:
        transforms_3d = T.Compose([instantiate(val['transform'])
                                   for val in conf.transforms_3d])
    return transforms, transforms_3d


def create_dataset(conf: DictConfig, title: str) -> Dataset:
    datasets = {
        'hdf5': HDF5Dataset,
        'images': ImagesDataset
    }
    frame_sampler = FrameSampler(conf.sample.frames,
                                 real_fake_ratio=conf.sample.real_fake_ratio,
                                 p_sparse=conf.sample.sparse_frames_prob)
    if conf.type not in datasets:
        known_types = ', '.join(map(str, datasets.keys()))
        raise AttributeError(
            "Unknown dataset type: {} in data.{}.type. "
            "Known types are: {}.".format(conf.type, title, known_types))
    DatasetImpl = datasets[conf.type]
    transforms, transforms_3d = create_transforms(conf)
    data = DatasetImpl(conf.dir,
                       frames=conf.sample.frames,
                       min_img_size=conf.sample.get('min_size', None),
                       sampler=frame_sampler,
                       transforms=transforms,
                       transforms_3d=transforms_3d,
                       sub_dirs=read_file_list(conf, title))
    print("Num {} samples: {}".format(title, len(data)))
    return data


def create_train_loader(conf: DictConfig, epoch_length=-1,
                        rank: Optional[int] = None,
                        num_replicas: Optional[int] = None) -> DataLoader:
    data = create_dataset(conf, 'train')
    bs = conf.loader.batch_size
    kwargs = dict()
    if epoch_length > 0:
        kwargs['num_samples'] = epoch_length * bs
    if num_replicas is not None and num_replicas > 1:
        if not isinstance(rank, int):
            raise AttributeError("Rank is missing")
        kwargs['rank'] = rank
        kwargs['num_replicas'] = num_replicas
    sampler = BalancedSampler(data, **kwargs)
    num_workers = conf.get('loader.workers', 0)
    loader = DataLoader(data, sampler=sampler, batch_size=bs,
                        num_workers=num_workers, drop_last=True)
    return loader


def create_val_loader(conf: DictConfig, rank: Optional[int] = None,
                      num_replicas: Optional[int] = None) -> DataLoader:
    data = create_dataset(conf, 'val')
    bs = conf.loader.batch_size
    num_workers = conf.get('loader.workers', 0)
    sampler = None
    if num_replicas is not None and num_replicas > 1:
        if not isinstance(rank, int):
            raise AttributeError("Rank is missing")
        sampler = DistributedSampler(data, rank=rank, num_replicas=num_replicas, shuffle=False)
    loader = DataLoader(data, sampler=sampler, batch_size=bs, num_workers=num_workers)
    return loader
