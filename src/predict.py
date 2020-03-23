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


@hydra.main(config_path="../config/predict.yaml")
def main(conf: DictConfig):
    print(conf.pretty())

    if 'gpu' in conf.general:
        torch.cuda.set_device(conf.general.gpu)
    device = torch.device('cuda')
    torch.manual_seed(conf.general.seed)

    model = instantiate(conf.model).to(device)
    state = torch.load(conf.model.weights)
    assert isinstance(state, dict)
    if 'model' in state.keys():
        state = state['model']
    model.load_state_dict(state)


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    main()
