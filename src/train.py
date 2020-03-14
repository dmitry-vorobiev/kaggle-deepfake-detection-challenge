import datetime as dt
import time
import torch
import torchvision.transforms as T
import sys
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import BatchSampler, DataLoader
from typing import Dict, Iterable, List, Tuple, Union

from dataset.hdf5 import HDF5Dataset
from dataset.images import ImagesDataset
from dataset.sample import FrameSampler, BalancedSampler
from model.detector import basic_detector_256, DetectorOut
from model.loss import combined_loss


HDF5_DIR = '/media/dmitry/other/dfdc-crops/hdf5'
Batch = Tuple[Tensor, Tensor]


def create_loader(bs: int, num_frames: int, real_fake_ratio: float,
                  p_sparse_frames: float, chunks: Iterable[int]) -> DataLoader:
    dirs = [f'dfdc_train_part_{i}' for i in chunks]
    sampler = FrameSampler(num_frames,
                           real_fake_ratio=real_fake_ratio,
                           p_sparse=p_sparse_frames)
    transforms = T.Compose([T.ToTensor()])

    ds = HDF5Dataset(HDF5_DIR,
                     size=(num_frames, 256),
                     sampler=sampler,
                     transforms=transforms,
                     sub_dirs=dirs)
    print('Num samples: {}'.format(len(ds)))

    batch_sampler = BatchSampler(BalancedSampler(ds), batch_size=bs, drop_last=True)
    return DataLoader(ds, batch_sampler=batch_sampler)


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
    print("\n[{}] {:>5} | ep: {}, acc: {:.3f}, nll: {:.3f}\n".format(
        cur_time, title, epoch, metrics['acc'], metrics['nll']))


def main():
    train_dl = create_loader(
        bs=12,
        num_frames=10,
        real_fake_ratio=100 / 30,
        p_sparse_frames=0.75,
        chunks=range(5, 30))

    valid_dl = create_loader(
        bs=12,
        num_frames=10,
        real_fake_ratio=100 / 30,
        p_sparse_frames=1.,
        chunks=range(0, 5))

    device = torch.device('cuda:1')
    model = basic_detector_256().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(engine: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.train()
        optim.zero_grad()
        x, y = prepare_batch(batch, device)
        out = model(x, y)
        loss = combined_loss(out, x, y)
        loss.backward()
        optim.step()
        return gather_outs(batch, out, loss)

    def validate(engine: Engine, batch: Batch) -> Dict[str, Tensor]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            out = model(x, y)
            loss = combined_loss(out, x, y)
        return gather_outs(batch, out, loss)

    trainer = Engine(train)
    evaluator = Engine(validate)

    def parse_y(out: Dict[str, Tensor]):
        return [out['y_pred'], out['y_true']]

    accuracy = Accuracy(output_transform=parse_y)
    log_loss = Loss(torch.nn.BCELoss(), output_transform=parse_y)

    for metric, key in zip([accuracy, log_loss], ['acc', 'nll']):
        for engine in [trainer, evaluator]:
            metric.attach(engine, key)

    log_interval = 1

    iter_complete = Events.ITERATION_COMPLETED(every=log_interval)

    for engine, name in zip([trainer, evaluator], ['train', 'val']):
        engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        engine.add_event_handler(iter_complete, log_iter, trainer, name, log_interval)
        engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, trainer, name)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(valid_dl, epoch_length=5))
    trainer.run(train_dl, max_epochs=3, epoch_length=10)


if __name__ == '__main__':
    main()
