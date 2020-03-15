import datetime as dt
import hydra
import os
import time
import torch
import torchvision.transforms as T

from functools import partial
from ignite.engine import Engine, Events, State
from ignite.metrics import Accuracy, Loss, Metric
from omegaconf import DictConfig, ListConfig
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import BatchSampler, DataLoader
from typing import Dict, Iterable, List, Tuple, Union

from dataset import HDF5Dataset, ImagesDataset, FrameSampler, BalancedSampler
from model.detector import FakeDetector, DetectorOut
from model.loss import combined_loss

Batch = Tuple[Tensor, Tensor]

datasets = {
    'hdf5': HDF5Dataset,
    'images': ImagesDataset
}


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


def create_loader(conf: DictConfig, img_size: int, title: str) -> DataLoader:
    num_frames = conf.sample.frames
    sampler = FrameSampler(num_frames,
                           real_fake_ratio=conf.sample.real_fake_ratio,
                           p_sparse=conf.sample.sparse_frames_prob)
    transforms = T.Compose([T.ToTensor()])
    if conf.type not in datasets:
        known_types = ', '.join(map(str, datasets.keys()))
        raise AttributeError(
            "Unknown dataset type: {} in data.{}.type. "
            "Known types are: {}.".format(conf.type, title, known_types))
    Dataset = datasets[conf.type]
    ds = Dataset(conf.dir,
                 size=(num_frames, img_size),
                 sampler=sampler,
                 transforms=transforms,
                 sub_dirs=read_file_list(conf, title))
    print("Num {} samples: {}".format(title, len(ds)))

    batch_sampler = BatchSampler(BalancedSampler(ds),
                                 batch_size=conf.loader.batch_size,
                                 drop_last=True)
    num_workers = conf.get('loader.workers', 0)
    return DataLoader(ds, batch_sampler=batch_sampler, num_workers=num_workers)


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


def log_iter(engine: Engine, trainer: Engine, title: str, log_interval: int) -> None:
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
    print("[{}][{:.2f} s] {:>5} | ep: {:2d}, it: {:3d}, {}".format(
        cur_time, it_time, title, epoch, iteration, stats))
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

    img_size = conf.model.img_size
    train_dl = create_loader(conf.data.train, img_size, 'train')
    valid_dl = create_loader(conf.data.val, img_size, 'val')

    device = create_device(conf)
    model = create_model(conf.model).to(device)
    optim = create_optimizer(conf.optimizer, model.parameters())

    metrics = {
        'acc': Accuracy(output_transform=_accuracy_transform),
        'nll': Loss(torch.nn.BCELoss(), output_transform=_nll_transform)
    }
    trainer = create_trainer(model, optim, device, metrics)
    evaluator = create_evaluator(model, device, metrics)

    log_interval = conf.logging.log_iter_interval
    iter_complete = Events.ITERATION_COMPLETED(every=log_interval)

    for engine, name in zip([trainer, evaluator], ['train', 'val']):
        engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        engine.add_event_handler(iter_complete, log_iter, trainer, name, log_interval)
        engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, trainer, name)

    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length
    if epoch_length < 1:
        epoch_length = len(train_dl)

    if 'schedule' in conf:
        lr_schedule = conf.schedule.strategy
        scheduler = None
        if lr_schedule == 'one-cycle':
            scheduler = create_one_cycle_scheduler(
                conf.schedule, epochs, epoch_length)(optim)
        elif lr_schedule == 'multi-step':
            print('Yo bro')
        else:
            raise AttributeError('Unknown schedule: {}'.format(conf.shedule))
        if scheduler:
            trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                      lambda _: scheduler.step())

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=conf.validate.interval),
        lambda _: evaluator.run(valid_dl, epoch_length=5))
    trainer.run(train_dl, max_epochs=epochs, epoch_length=epoch_length)


if __name__ == '__main__':
    main()
