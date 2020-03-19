import datetime as dt
import hydra
import logging
import os
import time
import torch

from functools import partial
from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss, Metric
from omegaconf import DictConfig, ListConfig
from torch import nn, FloatTensor, Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from dataset import HDF5Dataset, ImagesDataset, FrameSampler, BalancedSampler, simple_transforms
from model import ModelOut, TripleLoss

Batch = Tuple[Tensor, Tensor]
GatheredOuts = Dict[str, Union[float, Tensor]]
Metrics = Dict[str, Any]


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


def create_optimizer(conf: DictConfig, params: Iterable[FloatTensor]) -> torch.optim.Adam:
    optim = torch.optim.Adam(
        params,
        lr=conf.lr,
        betas=conf.betas,
        weight_decay=conf.weight_decay)
    return optim


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine: Engine, trainer: Engine, pbar: ProgressBar,
             title: str, log_interval: int) -> None:
    epoch = trainer.state.epoch
    iteration = engine.state.iteration
    out = engine.state.output
    stats = {k: '%.3f' % v for k, v in out.items() if 'loss' in k}
    if hasattr(engine.state, 'lr'):
        stats['lr'] = ', '.join(['%.1e' % val for val in engine.state.lr])
    stats = ', '.join(['{}: {}'.format(*e) for e in stats.items()])
    t0 = engine.state.t0
    t1 = time.time()
    it_time = (t1 - t0) / log_interval
    cur_time = humanize_time(t1)
    pbar.log_message("[{}][{:.2f} s] {:>5} | ep: {:2d}, it: {:3d}, {}".format(
        cur_time, it_time, title, epoch, iteration, stats))
    engine.state.t0 = t1


def log_epoch(engine: Engine, trainer: Engine, title: str) -> None:
    epoch = trainer.state.epoch
    metrics = engine.state.metrics
    logging.info("{:>5} | ep: {}, acc: {:.3f}, nll: {:.3f}\n".format(
        title, epoch, metrics['acc'], metrics['nll']))


def create_trainer(model: nn.Module, criterion: TripleLoss, optim: Any,
                   device: torch.device, metrics: Optional[Metrics] = None):
    def _update(engine: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.train()
        optim.zero_grad()
        x, y = prepare_batch(batch, device)
        out = model(x, y)
        losses = criterion(out, x, y)
        losses['loss'].backward()
        optim.step()
        engine.state.lr = [p['lr'] for p in optim.param_groups]
        return gather_outs(model, batch, out, losses)

    engine = Engine(_update)
    if metrics:
        add_metrics(engine, metrics)
    return engine


def create_evaluator(model: nn.Module, criterion: TripleLoss, device: torch.device,
                     metrics: Optional[Metrics] = None):
    def _eval(engine: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            out = model(x, y)
            losses = criterion(out, x, y)
        return gather_outs(model, batch, out, losses)

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


def gather_outs(model: nn.Module, batch: Batch, model_out: ModelOut,
                loss: Dict[str, Tensor]) -> GatheredOuts:
    out = {k: v.item() for k, v in loss.items()}
    out['y_pred'] = model.to_y(*model_out).detach().cpu()
    out['y_true'] = batch[-1].float().cpu()
    return out


def filter_losses(out: GatheredOuts) -> Dict[str, float]:
    losses = {k: v for k, v in out.items() if 'loss' in k}
    return losses


def _accuracy_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    return (out['y_pred'] >= 0.5).float(), out['y_true']


def _nll_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    return out['y_pred'], out['y_true']


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


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
    model = instantiate(conf.model).to(device)
    loss = instantiate(conf.loss)
    optim = create_optimizer(conf.optimizer, model.parameters())

    metrics = {
        'acc': Accuracy(output_transform=_accuracy_transform),
        'nll': Loss(torch.nn.BCELoss(), output_transform=_nll_transform)
    }
    trainer = create_trainer(model, loss, optim, device, metrics)
    evaluator = create_evaluator(model, loss, device, metrics)

    every_iteration = Events.ITERATION_COMPLETED

    if 'lr_scheduler' in conf.keys():
        # TODO: total_steps is wrong, it works only for one-cycle
        lr_scheduler = instantiate(
            conf.lr_scheduler, optim, total_steps=epoch_length)
        trainer.add_event_handler(every_iteration, lambda _: lr_scheduler.step())

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            initial_state = lr_scheduler.state_dict()
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=epoch_length),
                                      lambda _: lr_scheduler.load_state_dict(initial_state))
    else:
        lr_scheduler = None

    log_freq = conf.logging.iter_freq
    log_event = Events.ITERATION_COMPLETED(every=log_freq)
    pbar = ProgressBar(persist=False)

    for engine, name in zip([trainer, evaluator], ['train', 'val']):
        engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        engine.add_event_handler(log_event, log_iter, trainer, pbar, name, log_freq)
        engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, trainer, name)
        pbar.attach(engine, output_transform=filter_losses)

    trainer.add_event_handler(every_iteration, TerminateOnNan())

    cp = conf.train.checkpoints
    to_save = {
        'trainer': trainer,
        'model': model,
        'optimizer': optim,
        'lr_scheduler': lr_scheduler
    }

    if 'load' in cp.keys() and cp.load:
        trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)
        logging.info("Resume from a checkpoint: {}".format(cp.load))
        Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(cp.load))

    save_path = cp.get('base_dir', os.getcwd())
    logging.info("Saving checkpoints to {}".format(save_path))
    max_cp = max(int(cp.get('max_checkpoints', 1)), 1)
    save = DiskSaver(save_path, create_dir=True, require_empty=True)

    make_checkpoint = Checkpoint(to_save, save, n_saved=max_cp)
    cp_iter = cp.interval_iteration
    cp_epoch = cp.interval_epoch
    if cp_iter > 0:
        save_event = Events.ITERATION_COMPLETED(every=cp_iter)
        trainer.add_event_handler(save_event, make_checkpoint)
    if cp_epoch > 0:
        if cp_iter < 1 or epoch_length % cp_iter:
            save_event = Events.EPOCH_COMPLETED(every=cp_epoch)
            trainer.add_event_handler(save_event, make_checkpoint)

    eval_event = Events.EPOCH_COMPLETED(every=conf.validate.interval)
    trainer.add_event_handler(eval_event, lambda _: evaluator.run(valid_dl))

    try:
        trainer.run(train_dl, max_epochs=epochs)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    pbar.close()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
