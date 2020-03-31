import gc
import hydra
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import torchvision.transforms as T
import traceback

from functools import partial
from hydra.utils import instantiate
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from omegaconf import DictConfig
from torch import nn, Tensor
from typing import Any, Dict, List, Tuple

# TODO: make proper setup.py
sys.path.insert(0, f'/home/{os.environ["USER"]}/projects/dfdc/vendors/Pytorch_Retinaface')
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox

from image import crop_square_torch
from dataset.utils import pad_torch
from detection_utils import max_num_faces
from detectors.retinaface import init_detector, prepare_imgs, postproc_detections_gpu
from video import VideoPipe, parse_meta, read_frames_cv2


def write_file_list(files: List[str], path: str) -> None:
    with open(path, mode='w') as h:
        for i, f in enumerate(files):
            if os.path.isfile(f):
                h.write(f'{f} {i}\n')


def split_files_by_res(files_meta: np.ndarray, min_freq: int
                       ) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
    px_count = files_meta[:, 1] * files_meta[:, 2]
    clusters, freq = np.unique(px_count, return_counts=True)
    cluster_idxs, px_mults, heap = [], [], []

    for i, px_c in enumerate(clusters):
        idxs = (px_count == px_c).nonzero()[0]
        if freq[i] >= min_freq:
            cluster_idxs.append(idxs)
            px_m = float(px_c / (1920 * 1080))
            px_mults.append(px_m)
        else:
            heap.append(idxs)

    if len(heap) > 0:
        heap = np.hstack(heap)
        heap.sort()
    else:
        heap = np.empty(0, dtype=np.int64)
    return cluster_idxs, px_mults, heap


def merge_detector_cfg(conf: DictConfig) -> Dict[str, any]:
    cfg = cfg_mnet if conf.encoder == 'mnet' else cfg_re50
    for key in "batch_size, score_thresh, nms_thresh, top_k, keep_top_k".split(", "):
        if key not in conf or conf[key] is None:
            raise AttributeError("Missing {} in detector config".format(key))
    cfg = {**cfg, **conf}
    return cfg


def find_faces(frames: Tensor, model: torch.nn.Module,
               device: torch.device, conf: Dict[str, Any]) -> List[Tensor]:
    D, H, W, C = frames.shape
    frames_orig = frames.permute(0, 3, 1, 2)
    frames, scale = prepare_imgs(frames)
    prior_box = PriorBox(conf, image_size=(H, W))
    priors = prior_box.forward().to(device)
    scale = scale.to(device)

    with torch.no_grad():
        locations, confidence, landmarks = model(frames)
        detections = postproc_detections_gpu(
            locations, confidence, priors, scale, conf)

    num_faces = np.array(list(map(len, detections)), dtype=np.uint8)
    while (num_faces.mean() < conf['min_positive_rate'] and
           conf['score_thresh'] >= conf['score_thresh_min']):
        conf = dict(conf)
        conf['score_thresh'] -= conf['score_thresh_step']
        detections = postproc_detections_gpu(
            locations, confidence, priors, scale, conf)
        num_faces = np.array(list(map(len, detections)), dtype=np.uint8)
        logging.debug(
            "Rerun detection postprocessing with score_thresh={:.02f}, "
            "avg_pos_rate={:.02f}".format(conf['score_thresh'], num_faces.mean()))

    max_faces = max_num_faces(num_faces, conf['max_face_num_thresh'])
    del locations, confidence, landmarks, priors, prior_box, scale, frames

    faces = []
    for f in range(D):
        for bbox in detections[f][:max_faces]:
            face = crop_square_torch(frames_orig[f], bbox[:4])
            if face is not None:
                faces.append(face)
    del detections, frames_orig
    return faces


def load_model(conf: DictConfig):
    model = instantiate(conf.model)
    state = torch.load(conf.model.weights)
    if 'model' in state.keys():
        state = state['model']
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


@hydra.main(config_path="../config/predict.yaml")
def main(conf: DictConfig):
    print(conf.pretty())

    if 'gpu' in conf.general.keys():
        torch.cuda.set_device(conf.general.gpu)
    device = torch.device('cuda')
    torch.manual_seed(conf.general.seed)

    model = load_model(conf).to(device)

    face_det_conf = merge_detector_cfg(conf['face-detection'])
    detector = init_detector(face_det_conf, face_det_conf['weights'], device).to(device)
    crop_faces = partial(find_faces, model=detector, device=device, conf=face_det_conf)

    data_conf = conf.data.test
    base_dir = data_conf.dir
    file_names = [file for file in os.listdir(base_dir) if file.endswith('mp4')]
    files = [os.path.join(base_dir, file) for file in file_names]
    if not len(files):
        raise RuntimeError("No files was found in {}".format(base_dir))
    else:
        logging.info("Total number of files: {}".format(len(files)))
    logging.info("Parsing video metadata...")
    files_meta = parse_meta(files)
    splits, px_mults, leftover = split_files_by_res(files_meta, 5)
    px_mults_str = ' '.join(map(lambda f: '%.03f' % f, iter(px_mults)))
    logging.debug("Found {} clusters, px_mults are: {}. {} items are left over".format(
        len(splits), px_mults_str, len(leftover)))
    transforms = T.Compose([instantiate(val['transform']) for val in data_conf.transforms])
    logging.debug("Using transforms: {}".format(transforms))

    def _predict(images: List[Tensor]) -> float:
        last_idx = len(images) - 1
        num_samples = min(data_conf.sample.max_samples, len(images))
        idxs = np.linspace(0, last_idx, num_samples, dtype=int, endpoint=True)
        images = list(map(images.__getitem__, idxs))

        x = torch.stack(list(map(transforms, images)))
        pad_amount = reader_conf.frames - x.size(0)
        if pad_amount > 0:
            x = pad_torch(x, pad_amount, 'start')
        # D, C, H, W -> C, D, H, W
        x = x.transpose(0, 1).unsqueeze_(0)
        with torch.no_grad():
            out = model(x, None)
        y_pred = model.to_y(*out).cpu().item()
        return y_pred

    def _handle(images: List[Tensor], split_idx: int, pipe_name: str):
        glob_idx = split_idxs[split_idx]
        file_path = split_files[split_idx]
        if len(images) > 0:
            y_hat = _predict(images)
            df.loc[glob_idx, 'label'] = y_hat
            logging.info("{} | {} | faces: {} | y: {:.03f} | {}".format(
                pipe_name, glob_idx, len(images), y_hat, file_path))
        else:
            ignore_split_files[split_idx] = True
            logging.warning("No faces have found in ({}): {}".format(glob_idx, file_path))

    def _on_exception(e: Exception):
        nonlocal video_batch, frames, faces
        logging.error(traceback.format_exc())
        video_batch, frames = None, None
        faces = []
        gc.collect()

    def _save():
        save_dir = conf.get('general.save_dir', os.getcwd())
        save_path = os.path.join(save_dir, conf.general.csv_name)
        logging.info("Saving predictions to {}".format(save_path))
        for _ in range(5):
            try:
                df.to_csv(save_path, index=False)
                break
            except Exception as e:
                continue

    reader_conf = data_conf.sample
    df = pd.DataFrame(file_names, columns=['filename'])
    df['label'] = 0.5
    del file_names, files_meta

    for s, split_idxs in enumerate(splits):
        split_files = list(map(lambda i: files[i], split_idxs))
        logging.debug("Num files in cluster {}: {}".format(s, len(split_files)))
        handled_split_files = np.zeros(len(split_files), dtype=np.bool)
        ignore_split_files = np.zeros(len(split_files), dtype=np.bool)
        seq_len = int(reader_conf.frames / reader_conf.num_pass / max(px_mults[s], 1.0))
        stride = reader_conf.stride
        logging.debug(
            "Running predictions on cluster {} (size_factor={:.03f})".format(s, px_mults[s]))
        step = data_conf.max_open_files

        for offset in range(0, len(split_files), step):
            last = min(offset + step, len(split_files))
            logging.debug("Running chunk [{}:{}] from {} cluster".format(offset, last, s))
            write_file_list(split_files[offset:last], path='temp')
            logging.debug("Creating new pipe with seq_len={}, stride={}".format(seq_len, stride))
            pipe = VideoPipe('temp', seq_len=seq_len, stride=stride, device_id=0)
            pipe.build()
            num_samples_read = pipe.epoch_size('reader')
            logging.debug("Pipe length: {}".format(num_samples_read))

            if num_samples_read > 0:
                logging.debug("Creating DALI iterator...")
                data_iter = DALIGenericIterator(
                    [pipe], ['frames', 'label'], num_samples_read, dynamic_shape=True)
                video_batch, frames, prev_idx = None, None, None
                faces = []
                while True:
                    try:
                        video_batch = next(data_iter)
                        frames = video_batch[0]['frames'].squeeze(0)
                        read_idx = video_batch[0]['label'].item()
                        new_faces = crop_faces(frames)
                        del video_batch, frames

                        if prev_idx is None or prev_idx == read_idx:
                            faces += new_faces
                        else:
                            abs_idx = offset + prev_idx
                            _handle(faces, abs_idx, 'dali')
                            handled_split_files[abs_idx] = True
                            faces = new_faces
                        prev_idx = read_idx
                    except StopIteration:
                        break
                    except Exception as e:
                        _on_exception(e)
                # last sample in pipe
                try:
                    abs_idx = offset + prev_idx
                    _handle(faces, abs_idx, 'dali')
                    handled_split_files[abs_idx] = True
                except Exception as e:
                    _on_exception(e)

                del pipe, data_iter
                gc.collect()
                _save()
            logging.debug("Finished chunk [{}:{}] from {} cluster".format(offset, last, s))

        unhandled_files = (~handled_split_files).nonzero()[0]
        num_bad_samples = len(unhandled_files)
        if num_bad_samples > 0:
            logging.warning("Unable to read %d videos with DALI, running fallback "
                            "decoding through OpenCV..." % num_bad_samples)
            for idx in unhandled_files:
                abs_idx = split_idxs[idx]
                path = split_files[idx]
                if ignore_split_files[idx]:
                    logging.info("Ignoring file ({}): {}".format(abs_idx, path))
                    continue
                try:
                    frames = read_frames_cv2(path, reader_conf.frames)
                    if frames is not None and len(frames) > 0:
                        frames = torch.from_numpy(frames).to(device)
                        faces = crop_faces(frames)
                        del frames
                        _handle(faces, idx, 'cv2')
                        del faces
                except Exception as e:
                    _on_exception(e)
        del handled_split_files, unhandled_files, ignore_split_files
        _save()
        logging.debug("Finished cluster {} (size_factor={:.03f})".format(s, px_mults[s]))

    logging.info("Running predictions on leftover ({} samples)".format(len(leftover)))
    for idx in leftover:
        path = files[idx]
        try:
            frames = read_frames_cv2(path, reader_conf.frames)
            if frames is not None and len(frames) > 0:
                frames = torch.from_numpy(frames).to(device)
                faces = crop_faces(frames)
                del frames
                if len(faces) > 0:
                    y_hat = _predict(faces)
                    df.loc[idx, 'label'] = y_hat
                    logging.info("cv2 | {} | faces: {} | y: {:.03f} | {}".format(
                        idx, len(faces), y_hat, path))
                else:
                    logging.warning("No faces have found in ({}): {}".format(idx, path))
                del faces
        except Exception as e:
            _on_exception(e)
    logging.info("Finished leftover")

    _save()
    logging.info("DONE")


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    main()
