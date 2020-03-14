import datetime as dt
import hydra
import os
import time
import torch
import torchvision.transforms as T

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, Metric
from omegaconf import DictConfig, ListConfig
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import BatchSampler, DataLoader
from typing import Dict, Iterable, List, Tuple, Union

from dataset.hdf5 import HDF5Dataset
from dataset.images import ImagesDataset
from dataset.sample import FrameSampler, BalancedSampler
from model.detector import basic_detector_256, DetectorOut
from model.loss import combined_loss

Batch = Tuple[Tensor, Tensor]


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
                    "Config: incorrect format for 'loader.{}.chunks'".format(title))
        for c in chunks:
            dirs.append('dfdc_train_part_{}'.format(c))
    if 'dir_list' in conf.keys() and len(conf.dir_list):
        with open(conf.dir_list) as h:
            path = h.readline()
            if os.path.isdir(path):
                dirs.append(path)
    return dirs


def create_loader(conf: DictConfig, title: str) -> DataLoader:
    num_frames = conf.frames
    sampler = FrameSampler(num_frames,
                           real_fake_ratio=conf.real_fake_ratio,
                           p_sparse=conf.sparse_frames_prob)
    transforms = T.Compose([T.ToTensor()])
    Dataset = HDF5Dataset if conf.type == 'hdf5' else ImagesDataset
    ds = Dataset(conf.dir,
                 size=(num_frames, 256),
                 sampler=sampler,
                 transforms=transforms,
                 sub_dirs=read_file_list(conf, title))
    print('Num {} samples: {}'.format(title, len(ds)))

    batch_sampler = BatchSampler(BalancedSampler(ds),
                                 batch_size=conf.batch_size,
                                 drop_last=True)
    return DataLoader(ds, batch_sampler=batch_sampler)


def create_device(conf: DictConfig) -> torch.device:
    if 'gpu' not in conf.general.keys():
        return torch.device('cpu')
    gpu = conf.general.gpu
    if isinstance(gpu, ListConfig):
        gpu = gpu[0]
    return torch.device('cuda:{}'.format(gpu))


def prepare_batch(batch: Batch, device: torch.device) -> Batch:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return x, y


def gather_outs(batch: Batch, model_out: DetectorOut,
                loss: FloatTensor) -> Dict[str, Tensor]:
    y_pred = (model_out[-1] >= 0.5).flatten().float().detach().cpu()
    y_true = batch[-1].float().cpu()
    out = {'loss': loss.item(), 'y_pred': y_pred, 'y_true': y_true}
    return out


def _metrics_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    return out['y_pred'], out['y_true']


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine: Engine, trainer: Engine, title: str, log_interval: int) -> None:
    epoch = trainer.state.epoch
    iteration = engine.state.iteration
    loss = engine.state.output['loss']
    t0 = engine.state.t0
    t1 = time.time()
    it_time = (t1 - t0) / log_interval
    cur_time = humanize_time(t1)
    print("[{}][{:.2f} s] {:>5} | ep: {:2d}, it: {:3d}, loss: {:.5f}".format(
        cur_time, it_time, title, epoch, iteration, loss))
    engine.state.t0 = t1


def log_epoch(engine: Engine, trainer: Engine, title: str) -> None:
    epoch = trainer.state.epoch
    metrics = engine.state.metrics
    t1 = time.time()
    cur_time = humanize_time(t1)
    print("[{}] {:>5} | ep: {}, acc: {:.3f}, nll: {:.3f}\n".format(
        cur_time, title, epoch, metrics['acc'], metrics['nll']))


def create_trainer(model, optim, device, metrics=None):
    def _update(engine: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.train()
        optim.zero_grad()
        x, y = prepare_batch(batch, device)
        out = model(x, y)
        loss = combined_loss(out, x, y)
        loss.backward()
        optim.step()
        return gather_outs(batch, out, loss)

    engine = Engine(_update)
    if metrics:
        add_metrics(engine, metrics)
    return engine


def create_evaluator(model, device, metrics=None):
    def _eval(engine: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            out = model(x, y)
            loss = combined_loss(out, x, y)
        return gather_outs(batch, out, loss)

    engine = Engine(_eval)
    if metrics:
        add_metrics(engine, metrics)
    return engine


def add_metrics(engine: Engine, metrics: Dict[str, Metric]):
    for name, metric in metrics.items():
        metric._output_transform = _metrics_transform
        metric.attach(engine, name)


@hydra.main(config_path="../config/core.yaml")
def main(conf: DictConfig):
    print(conf.pretty())

    train_dl = create_loader(conf.loader.train, 'train')
    valid_dl = create_loader(conf.loader.val, 'val')

    device = create_device(conf)
    model = basic_detector_256().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=conf.optimizer.lr)

    metrics = {'acc': Accuracy(), 'nll': Loss(torch.nn.BCELoss())}
    trainer = create_trainer(model, optim, device, metrics)
    evaluator = create_evaluator(model, device, metrics)

    log_interval = conf.logging.log_iter_interval
    iter_complete = Events.ITERATION_COMPLETED(every=log_interval)

    for engine, name in zip([trainer, evaluator], ['train', 'val']):
        engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        engine.add_event_handler(iter_complete, log_iter, trainer, name, log_interval)
        engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, trainer, name)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(valid_dl, epoch_length=5))

    epoch_length = conf.train.epoch_length
    if epoch_length < 1:
        epoch_length = None
    trainer.run(train_dl, max_epochs=conf.train.epochs, epoch_length=epoch_length)


if __name__ == '__main__':
    main()
