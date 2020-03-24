import datetime as dt
import gc
import hydra
import logging
import numpy as np
import os
import sys
import time
import torch
import torchvision.transforms as T

from functools import partial
from hydra.utils import instantiate
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from omegaconf import DictConfig
from torch import nn, FloatTensor, Tensor
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# TODO: make proper setup.py
sys.path.insert(0, f'/home/{os.environ["USER"]}/projects/dfdc/vendors/Pytorch_Retinaface')
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox

from file_utils import write_file_list
from image import crop_square_torch
from dataset.utils import pad_torch
from detection_utils import max_num_faces
from detectors.retinaface import init_detector, prepare_imgs, postproc_detections_gpu
from video import VideoPipe, parse_meta, read_frames_cv2, split_files_by_res


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
    del locations, confidence, landmarks, priors, prior_box, scale, frames

    num_faces = np.array(list(map(len, detections)), dtype=np.uint8)
    max_faces = max_num_faces(num_faces, conf['max_face_num_thresh'])
    faces = []
    for f in range(D):
        for bbox in detections[f][:max_faces]:
            face = crop_square_torch(frames_orig[f], bbox[:4])
            if face is not None:
                faces.append(face)
    del detections, frames_orig
    return faces


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
    crop_faces = partial(find_faces, model=detector, device=device, conf=face_det_conf)

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
    transforms = T.Compose([instantiate(val['transform']) for val in data_conf.transforms])
    print("Using transforms: {}".format(transforms))

    def predict(images: List[Tensor]) -> float:
        x = torch.stack(list(map(transforms, images)))
        pad_amount = reader_conf.frames - x.size(0)
        if pad_amount > 0:
            x = pad_torch(x, pad_amount, 'start')
        # D, C, H, W -> C, D, H, W
        x = x.transpose(0, 1).unsqueeze_(0)
        out = model(x, None)
        y_hat = model.to_y(*out).cpu().numpy()
        return y_hat

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
                data_iter = DALIGenericIterator(
                    [pipe], ['frames', 'label'], num_samples_read, dynamic_shape=True)
                prev_idx = None
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
                            split_idx = offset + read_idx
                            abs_idx = split_idxs[split_idx]
                            y_pred = predict(faces)
                            logging.info("dali | {} | faces: {} | y: {} | {}".format(
                                abs_idx, len(faces), y_pred, files[abs_idx]))
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
                    frames = torch.from_numpy(frames).to(device)
                    faces = crop_faces(frames)
                    y_pred = predict(faces)
                    logging.info("cv2 | {} | faces: {} | y: {} | {}".format(
                        abs_idx, len(faces), y_pred, files[abs_idx]))


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    main()
