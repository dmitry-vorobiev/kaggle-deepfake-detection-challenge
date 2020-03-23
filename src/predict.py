import datetime as dt
import gc
import hydra
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist

from functools import partial
from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss, Metric, RunningAverage
from ignite.utils import convert_tensor
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from omegaconf import DictConfig
from torch import nn, FloatTensor, Tensor
from torch.nn.parallel import DistributedDataParallel
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# TODO: make proper setup.py
sys.path.insert(0, f'/home/{os.environ["USER"]}/projects/dfdc/vendors/Pytorch_Retinaface')
from data import cfg_mnet, cfg_re50

from file_utils import write_file_list
from image import crop_square
from detectors.retinaface import init_detector, detect
from video import VideoPipe, parse_meta, read_frames_cv2, split_files_by_res


def merge_detector_cfg(conf: DictConfig) -> Dict[str, any]:
    cfg = cfg_mnet if conf.encoder == 'mnet' else cfg_re50
    for key in "batch_size, score_thresh, nms_thresh, top_k, keep_top_k".split(", "):
        if key not in conf or conf[key] is None:
            raise AttributeError("Missing {} in detector config".format(key))
    cfg = {**cfg, **conf}
    return cfg


def find_faces(frames: np.ndarray, detect_fn: Callable,
               conf: Dict[str, Any]) -> List[np.ndarray]:
    max_face_num_thresh = conf['max_face_num_thresh']
    detections = detect_fn(frames)
    if isinstance(frames, Tensor):
        frames = frames.cpu().numpy()
    num_faces = np.array(list(map(len, detections)), dtype=np.uint8)
    max_faces = max_num_faces(num_faces, max_face_num_thresh)
    faces = []
    for f in range(len(frames)):
        for det in detections[f][:max_faces]:
            face = crop_square(frames[f], det[:4])
            if face is not None:
                faces.append(face)
    del detections
    return faces


def max_num_faces(face_counts: np.ndarray, unique_fraction_thresh: float) -> int:
    unique_values, unique_freq = np.unique(face_counts, return_counts=True)
    mask = unique_freq / len(face_counts) > unique_fraction_thresh
    return unique_values[mask].max()


@hydra.main(config_path="../config/predict.yaml")
def main(conf: DictConfig):
    print(conf.pretty())
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

    face_det_conf = merge_detector_cfg(conf['face-detection'])
    detector = init_detector(face_det_conf, face_det_conf['weights'], device).to(device)
    detect_fn = partial(detect, model=detector, cfg=face_det_conf, device=device)

    data_conf = conf.data.test
    base_dir = data_conf.dir
    files = [os.path.join(base_dir, file)
             for file in os.listdir(base_dir)
             if not file.endswith('.json')]
    if not len(files):
        raise RuntimeError('No files was found in {}'.format(base_dir))
    logging.info("Parsing video metadata...")
    files_meta = parse_meta(files)
    min_unique_res_freq = int(len(files) * 0.02)
    splits, size_factors = split_files_by_res(files_meta, min_unique_res_freq)
    size_factors_str = ' '.join(map(str, iter(size_factors)))
    logging.info(
        "Splitting files by resolution. Found {} clusters, "
        "size multipliers are: {}".format(len(size_factors), size_factors_str))

    reader_conf = data_conf.sample

    for s, mask in enumerate(splits):
        split_idxs = mask.nonzero()[0]
        handled_split_files = np.zeros(len(split_idxs), dtype=np.bool)
        seq_len = int(reader_conf.frames / reader_conf.num_pass / size_factors[s])
        step = data_conf.max_open_files

        for offset in range(0, len(mask), step):
            last = min(offset + step, len(mask))
            write_file_list(files[offset:last], path='temp', mask=mask)
            pipe = VideoPipe('temp', seq_len=seq_len, stride=reader_conf.stride, device_id=0)
            pipe.build()
            num_samples_read = pipe.epoch_size('reader')

            if num_samples_read > 0:
                data_iter = DALIGenericIterator([pipe], ['frames', 'label'], num_samples_read,
                                                dynamic_shape=True)
                prev_idx = None
                faces = []
                while True:
                    try:
                        video_batch = next(data_iter)
                        frames = video_batch[0]['frames'].squeeze(0)
                        read_idx = video_batch[0]['label'].item()
                        new_faces = find_faces(frames, detect_fn, face_det_conf)
                        del video_batch, frames

                        if prev_idx is None or prev_idx == read_idx:
                            faces += new_faces
                        else:
                            split_idx = offset + read_idx
                            abs_idx = split_idxs[split_idx]
                            logging.info("{} | faces: {} | {}".format(
                                abs_idx, len(faces), files[abs_idx]))
                            # run prediction
                            handled_split_files[split_idx] = True
                            faces = new_faces
                        prev_idx = read_idx
                    except StopIteration:
                        break
                    except Exception as e:
                        import traceback
                        logging.error(traceback.format_exc())
                        video_batch, frames = None, None
                        gc.collect()
                del pipe, data_iter
                gc.collect()

        unhandled_files = (~handled_split_files).nonzero()[0]
        num_bad_samples = len(unhandled_files)
        if num_bad_samples > 0:
            print("Unable to read %d videos with DALI\n"
                  "Running fallback decoding through OpenCV..." % num_bad_samples)
            for idx in unhandled_files:
                abs_idx = split_idxs[idx]
                frames = read_frames_cv2(files[abs_idx], reader_conf.frames)
                if frames is not None:
                    faces = find_faces(frames, detect_fn, face_det_conf)
                    logging.info("{} | faces: {} | {}".format(
                        abs_idx, len(faces), files[abs_idx]))
                    # run prediction


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    main()
