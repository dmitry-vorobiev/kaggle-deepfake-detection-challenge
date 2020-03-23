import datetime as dt
import hydra
import logging
import os
import time
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import convert_tensor
from omegaconf import DictConfig
from torch import nn, FloatTensor, Tensor
from torch.nn.parallel import DistributedDataParallel
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from model import ModelOut
from utils import create_train_loader, create_val_loader

Batch = Tuple[Tensor, Tensor]
Losses = Dict[str, Tensor]
Criterion = Callable[[ModelOut, Batch], Losses]
GatheredOuts = Dict[str, Union[float, Tensor]]
Metrics = Dict[str, Metric]


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine: Engine, trainer: Engine, pbar: ProgressBar,
             title: str, log_interval: int) -> None:
    epoch = trainer.state.epoch
    iteration = engine.state.iteration
    metrics = engine.state.metrics
    stats = {k: '%.3f' % v for k, v in metrics.items() if 'loss' in k}
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
    stats = {k: '%.3f' % metrics[k] for k in ['acc', 'nll']}
    stats = ', '.join(['{}: {}'.format(*e) for e in stats.items()])
    logging.info("{:>5} | ep: {}, {}".format(title, epoch, stats))


def create_trainer(model: nn.Module, criterion: Criterion, optim: Any,
                   device: torch.device, conf,
                   metrics: Optional[Metrics] = None):
    update_freq = conf.optimizer.step_interval

    def _update(e: Engine, batch: Batch) -> Dict[str, Tensor]:
        iteration = e.state.iteration
        model.train()
        batch = _prepare_batch(batch, device, non_blocking=True)
        out = model(*batch)
        losses = criterion(out, batch)

        if not iteration % update_freq:
            optim.zero_grad()
        losses['loss'].backward()
        if not (iteration + 1) % update_freq:
            optim.step()

        engine.state.lr = [p['lr'] for p in optim.param_groups]
        return gather_outs(model, batch, out, losses)

    engine = Engine(_update)
    if metrics:
        add_metrics(engine, metrics)
    return engine


def create_evaluator(model: nn.Module, criterion: Criterion,
                     device: torch.device, metrics: Optional[Metrics] = None):
    def _eval(e: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.eval()
        with torch.no_grad():
            batch = _prepare_batch(batch, device, non_blocking=True)
            outs = model(*batch)
            losses = criterion(outs, batch)
        return gather_outs(model, batch, outs, losses)

    engine = Engine(_eval)
    if metrics:
        add_metrics(engine, metrics)
    return engine


def add_metrics(engine: Engine, metrics: Metrics):
    for name, metric in metrics.items():
        metric.attach(engine, name)


def _prepare_batch(batch: Batch, device: torch.device,
                   non_blocking: bool) -> Batch:
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def gather_outs(model: nn.Module, batch: Batch, model_out: ModelOut,
                loss: Dict[str, Tensor]) -> GatheredOuts:
    out = dict(loss)
    out['y_pred'] = model.to_y(*model_out).detach().cpu()
    out['y_true'] = batch[-1].float().cpu()
    return out


def create_metrics(keys: List[str], device: Optional[torch.device] = None) -> Metrics:
    def _acc_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return (out['y_pred'] >= 0.5).float(), out['y_true']

    def _nll_transform(out: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return out['y_pred'], out['y_true']

    def _out_transform(key: str):
        return lambda out: out[key]

    metrics = {key: RunningAverage(output_transform=_out_transform(key))
               for key in keys}
    metrics['acc'] = Accuracy(output_transform=_acc_transform)
    metrics['nll'] = Loss(torch.nn.BCELoss(), output_transform=_nll_transform)
    if device:
        for m in metrics.values():
            m.device = device
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def run(conf: DictConfig):
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length

    dist_conf = conf.distributed
    local_rank = dist_conf.local_rank
    backend = dist_conf.backend
    distributed = backend is not None

    if local_rank == 0:
        print(conf.pretty())
    if distributed:
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        num_replicas = dist.get_world_size()
        epoch_length = epoch_length // num_replicas
        loader_args = dict(rank=rank, num_replicas=num_replicas)
    else:
        rank = 0
        torch.cuda.set_device(conf.general.gpu)
        loader_args = dict()

    device = torch.device('cuda')
    torch.manual_seed(conf.general.seed)

    train_dl = create_train_loader(conf.data.train, epoch_length=epoch_length, **loader_args)
    valid_dl = create_val_loader(conf.data.val, **loader_args)

    if epoch_length < 1:
        epoch_length = len(train_dl)

    model = instantiate(conf.model).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank, ], output_device=local_rank)
        model.to_y = model.module.to_y
    if rank == 0 and conf.logging.model:
        print(model)

    loss = instantiate(conf.loss)
    optim = instantiate(conf.optimizer, model.parameters())
    metrics = create_metrics(loss.keys(), device if distributed else None)
    trainer = create_trainer(model, loss, optim, device, conf, metrics)
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

    trainer.add_event_handler(every_iteration, TerminateOnNan())

    cp = conf.train.checkpoints
    to_save = {
        'trainer': trainer,
        'model': model.module if distributed else model,
        'optimizer': optim,
        'lr_scheduler': lr_scheduler
    }

    if rank == 0:
        log_freq = conf.logging.iter_freq
        log_event = Events.ITERATION_COMPLETED(every=log_freq)
        pbar = ProgressBar(persist=False)

        for engine, name in zip([trainer, evaluator], ['train', 'val']):
            engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
            engine.add_event_handler(log_event, log_iter, trainer, pbar, name, log_freq)
            engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, trainer, name)
            pbar.attach(engine, metric_names=loss.keys())

        if 'load' in cp.keys() and cp.load:
            logging.info("Resume from a checkpoint: {}".format(cp.load))
            trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)

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

    if 'load' in cp.keys() and cp.load:
        Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(cp.load))

    sampler = train_dl.sampler
    assert sampler is not None
    trainer.add_event_handler(
        Events.EPOCH_STARTED, lambda e: sampler.set_epoch(e.state.epoch - 1))

    def run_validation(e: Engine):
        torch.cuda.synchronize(device)
        evaluator.run(valid_dl)

    eval_event = Events.EPOCH_COMPLETED(every=conf.validate.interval)
    trainer.add_event_handler(eval_event, run_validation)

    try:
        trainer.run(train_dl, max_epochs=epochs)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    if rank == 0:
        pbar.close()


@hydra.main(config_path="../config/train.yaml")
def main(conf: DictConfig):
    dist_conf = conf.distributed
    local_rank = dist_conf.local_rank
    backend = dist_conf.backend
    distributed = backend is not None

    if distributed:
        dist.init_process_group(backend, init_method=dist_conf.url)
        if local_rank == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}".format(dist.get_rank()))
            print("\n")

    try:
        run(conf)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        if distributed:
            dist.destroy_process_group()
        raise e

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    main()
