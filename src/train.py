import datetime as dt
import hydra
import os
import time
import torch

from functools import partial
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, Metric
from omegaconf import DictConfig, ListConfig
from torch import FloatTensor, Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from dataset import HDF5Dataset, ImagesDataset, FrameSampler, BalancedSampler, simple_transforms
from model.detector import FakeDetector, DetectorOut
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
                    "Config: incorrect format for 'data.{}.chunks'".format(title))
        for c in chunks:
            dirs.append('dfdc_train_part_{}'.format(c))
    if 'dir_list' in conf.keys() and len(conf.dir_list):
        with open(conf.dir_list) as h:
            path = h.readline()
            if os.path.isdir(path):
                dirs.append(path)
    return dirs


def create_transforms(conf: DictConfig) -> Callable[[Any], Tensor]:
    transforms = simple_transforms(
        conf.resize_to,
        mean=conf.mean,
        std=conf.std,
        hpf_n=conf.hpf_order)
    return transforms


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
    data = DatasetImpl(conf.dir,
                       frames=conf.sample.frames,
                       sampler=frame_sampler,
                       transforms=create_transforms(conf.transforms),
                       sub_dirs=read_file_list(conf, title))
    print("Num {} samples: {}".format(title, len(data)))
    return data


def create_loader(conf: DictConfig, title: str, epoch_length=-1) -> DataLoader:
    data = create_dataset(conf, title)
    bs = conf.loader.batch_size
    if epoch_length > 0:
        num_samples = epoch_length * bs
        item_sampler = BalancedSampler(data, replacement=True, num_samples=num_samples)
    else:
        item_sampler = BalancedSampler(data)
    batch_sampler = BatchSampler(item_sampler, batch_size=bs, drop_last=True)
    loader = DataLoader(
        data, batch_sampler=batch_sampler, num_workers=conf.get('loader.workers', 0))
    return loader


def create_device(conf: DictConfig) -> torch.device:
    if 'gpu' not in conf.general.keys():
        return torch.device('cpu')
    gpu = conf.general.gpu
    if isinstance(gpu, ListConfig):
        gpu = gpu[0]
    return torch.device('cuda:{}'.format(gpu))


def create_model(conf: DictConfig) -> FakeDetector:
    model = FakeDetector(
        img_size=conf.img_size,
        enc_depth=conf.enc_depth,
        enc_width=conf.enc_width,
        mid_layers=list(conf.mid_layers),
        out_ch=conf.out_ch)
    return model


def create_optimizer(conf: DictConfig, params: Iterable[FloatTensor]) -> torch.optim.Adam:
    optim = torch.optim.Adam(
        params,
        lr=conf.lr,
        betas=conf.betas,
        weight_decay=conf.weight_decay)
    return optim


def create_one_cycle_scheduler(conf: DictConfig, epochs: int, epoch_length: int):
    scheduler = partial(
        torch.optim.lr_scheduler.OneCycleLR,
        max_lr=conf.max_lr,
        epochs=epochs,
        steps_per_epoch=epoch_length,
        pct_start=conf.pct_start,
        anneal_strategy=conf.anneal_strategy,
        base_momentum=conf.base_momentum,
        max_momentum=conf.max_momentum)
    return scheduler


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine: Engine, trainer: Engine, pbar: ProgressBar,
             title: str, log_interval: int) -> None:
    epoch = trainer.state.epoch
    iteration = engine.state.iteration
    stats = dict()
    if hasattr(engine.state, 'lr'):
        stats['lr'] = ', '.join(['%.1e' % val for val in engine.state.lr])
    stats['loss'] = '%.3f' % engine.state.output['loss']
    stats = ', '.join(['{}: {}'.format(*e) for e in stats.items()])
    t0 = engine.state.t0
    t1 = time.time()
    it_time = (t1 - t0) / log_interval
    cur_time = humanize_time(t1)
    pbar.log_message("[{}][{:.2f} s] {:>5} | ep: {:2d}, it: {:3d}, {}".format(
        cur_time, it_time, title, epoch, iteration, stats))
    engine.state.t0 = t1


def log_epoch(engine: Engine, trainer: Engine, pbar: ProgressBar, title: str) -> None:
    epoch = trainer.state.epoch
    metrics = engine.state.metrics
    t1 = time.time()
    cur_time = humanize_time(t1)
    pbar.log_message("[{}] {:>5} | ep: {}, acc: {:.3f}, nll: {:.3f}\n".format(
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
        engine.state.lr = [p['lr'] for p in optim.param_groups]
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
        metric.attach(engine, name)


def prepare_batch(batch: Batch, device: torch.device) -> Batch:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return x, y


def gather_outs(batch: Batch, model_out: DetectorOut,
                loss: FloatTensor) -> Dict[str, Tensor]:
    y_pred = model_out[-1].detach()
    y_pred = torch.sigmoid(y_pred).squeeze_(1).cpu()
    y_true = batch[-1].float().cpu()
    out = {'loss': loss.item(), 'y_pred': y_pred, 'y_true': y_true}
    return out


def _accuracy_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    return (out['y_pred'] >= 0.5).float(), out['y_true']


def _nll_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    return out['y_pred'], out['y_true']


@hydra.main(config_path="../config/core.yaml")
def main(conf: DictConfig):
    print(conf.pretty())
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length

    train_dl = create_loader(conf.data.train, 'train', epoch_length=epoch_length)
    valid_dl = create_loader(conf.data.val, 'val')

    if epoch_length < 1:
        epoch_length = len(train_dl)

    device = create_device(conf)
    model = create_model(conf.model).to(device)
    optim = create_optimizer(conf.optimizer, model.parameters())

    lr_scheduler = None
    if 'lr_schedule' in conf:
        policy = conf.lr_schedule.strategy
        if policy == 'one-cycle':
            lr_scheduler = create_one_cycle_scheduler(
                conf.lr_schedule, epochs, epoch_length)(optim)
        elif policy == 'multi-step':
            raise NotImplementedError()
        else:
            raise AttributeError('Unknown lr_schedule: {}'.format(conf.shedule))

    metrics = {
        'acc': Accuracy(output_transform=_accuracy_transform),
        'nll': Loss(torch.nn.BCELoss(), output_transform=_nll_transform)
    }
    trainer = create_trainer(model, optim, device, metrics)
    evaluator = create_evaluator(model, device, metrics)

    log_interval = conf.logging.log_iter_interval
    iter_complete = Events.ITERATION_COMPLETED(every=log_interval)

    pbar = ProgressBar(persist=False)
    pbar.attach(trainer, output_transform=lambda out: {'loss': out['loss']})

    for engine, name in zip([trainer, evaluator], ['train', 'val']):
        engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        engine.add_event_handler(iter_complete, log_iter, trainer, pbar, name, log_interval)
        engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, trainer, pbar, name)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=conf.validate.interval),
        lambda _: evaluator.run(valid_dl, epoch_length=5))

    if lr_scheduler:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda _: lr_scheduler.step())

    to_save = {
        'trainer': trainer,
        'model': model,
        'optimizer': optim,
        'lr_scheduler': lr_scheduler}
    make_checkpoint = Checkpoint(to_save, DiskSaver('/tmp/training', create_dir=True))
    save_events = []
    cp = conf.train.checkpoints
    if cp.interval_iteration > 0:
        save_events.append(Events.ITERATION_COMPLETED(every=cp.interval_iteration))
    if cp.interval_epoch > 0:
        save_events.append(Events.EPOCH_COMPLETED(every=cp.interval_epoch))
    for event in save_events:
        trainer.add_event_handler(event, make_checkpoint)

    trainer.run(train_dl, max_epochs=epochs, epoch_length=epoch_length)


if __name__ == '__main__':
    main()
