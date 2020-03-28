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
from omegaconf import DictConfig
from torch import Tensor
from typing import Any, Dict, List, Tuple

# TODO: make proper setup.py
sys.path.insert(0, f'/home/{os.environ["USER"]}/projects/dfdc/vendors/Pytorch_Retinaface')
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox

from image import crop_square_torch
from dataset.utils import pad_torch
from detection_utils import max_num_faces
from detectors.retinaface import init_detector, prepare_imgs, postproc_detections_gpu
from predict import merge_detector_cfg, load_model
from video import read_frames_cv2


def find_faces(frames: Tensor, model: torch.nn.Module,
               device: torch.device, conf: Dict[str, Any]) -> List[Tensor]:
    D, H, W, C = frames.shape
    # D, H, W, C -> D, C, H, W
    frames_orig = frames.permute(0, 3, 1, 2)
    frames, scale = prepare_imgs(frames)
    prior_box = PriorBox(conf, image_size=(H, W))
    priors = prior_box.forward().to(device)
    scale = scale.to(device)

    chunk_size = conf["batch_size"]
    detections = []
    for start in range(0, D, chunk_size):
        end = start + chunk_size
        with torch.no_grad():
            locations, confidence, landmarks = model(frames[start:end])
            del landmarks
        det_chunk = postproc_detections_gpu(
            locations, confidence, priors, scale, conf)
        detections.extend(det_chunk)
        del locations, confidence
    del priors, prior_box, scale, frames

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

    def _save():
        save_dir = conf.get('general.save_dir', os.getcwd())
        save_path = os.path.join(save_dir, conf.general.csv_name)
        logging.info("Saving predictions to {}".format(save_path))
        for _ in range(5):
            try:
                df.to_csv(save_path, index=False)
                break
            except Exception as e:
                logging.error(traceback.format_exc())
                continue

    reader_conf = data_conf.sample
    df = pd.DataFrame(file_names, columns=['filename'])
    df['label'] = 0.5
    del file_names

    for idx, path in enumerate(files):
        if not (idx + 1) % 100:
            _save()
            gc.collect()
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
            logging.error(traceback.format_exc())
            torch.cuda.empty_cache()
            gc.collect()
    _save()
    logging.info("DONE")


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    main()
