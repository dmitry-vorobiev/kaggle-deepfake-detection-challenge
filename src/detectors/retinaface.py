import math
import numba
import numpy as np
from functools import partial
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms


@numba.njit
def adjust_bs(bs: int, height: int, width: int) -> int:
    pixels = width * height
    # full_hd = 1, quad_hd = 4
    down_ratio = math.ceil(pixels / 2073600)**2
    return bs // down_ratio

# @numba.njit
# def adjust_bs(bs: int, height: int, width: int) -> int:
#     pixels = width * height
#     # full_hd = 1, quad_hd = 3
#     down_ratio = math.ceil((pixels / 2073600) - 1) * 2 + 1
#     return bs // down_ratio


def detect(sample: Union[np.ndarray, Tensor], model: torch.nn.Module, 
           cfg: Dict[str,any], device: torch.device) -> List[np.ndarray]:
    num_frames, height, width, ch = sample.shape
    bs = cfg['batch_size']
    bs = adjust_bs(bs, height, width)
    imgs, scale = prepare_imgs(sample)
    
    priorbox = PriorBox(cfg, image_size=(height, width))
    priors = priorbox.forward().to(device)
    scale = scale.to(device)
    detections = []
    
    for start in range(0, num_frames, bs):
        end = start + bs
        imgs_batch = imgs[start:end].to(device)
        with torch.no_grad():
            loc, conf, landms = model(imgs_batch)
        imgs_batch, landms = None, None
        dets = postproc_detections(loc, conf, priors, scale, cfg)
        detections.extend(dets)
        loc, conf = None, None
    return detections


def prepare_imgs(sample: Union[np.ndarray, Tensor]) -> Tuple[Tensor, Tensor]:
    n, h, w, c = sample.shape
    mean = [104, 117, 123]
    if isinstance(sample, Tensor):
        imgs = sample.float()
        imgs -= torch.tensor(mean, device=imgs.device)
        imgs = imgs.permute(0, 3, 1, 2)
    else:
        imgs = np.float32(sample)
        imgs -= mean
        imgs = imgs.transpose(0, 3, 1, 2)
        imgs = torch.from_numpy(imgs)
    scale = torch.tensor([w, h, w, h])
    return imgs, scale


def postproc_detections(
        locations: Tensor, confidence: Tensor, priors: Tensor, 
        scale: Tensor, cfg: Dict[str, any], resize=1) -> List[np.ndarray]:
    boxes = decode_batch(locations, priors, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = confidence.cpu().numpy()[:, :, 1]
    num_frames = scores.shape[0]
    proc_fn = partial(postproc_frame, 
        score_thresh=cfg['score_thresh'], 
        nms_thresh=cfg['nms_thresh'], 
        top_k=cfg['top_k'], 
        keep_top_k=cfg['keep_top_k'])
    dets = [proc_fn(boxes[i], scores[i]) for i in range(num_frames)]
    return dets


def decode_batch(loc: Tensor, priors: Tensor, variances) -> Tensor:
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [n_samples, num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def postproc_frame(
        boxes: np.ndarray, scores: np.ndarray, 
        score_thresh=0.75, nms_thresh=0.4, 
        top_k=500, keep_top_k=5) -> np.ndarray:
    inds = (scores > score_thresh).nonzero()[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis]))
    dets = dets.astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_thresh)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    return dets


def init_detector(cfg: Dict[str, any], weights: str, device: torch.device) -> torch.nn.Module:
    cfg['pretrain'] = False
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, weights, device)
    net.eval()
    return net


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters 
        sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device=None):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if device:
        pretrained_dict = torch.load(
            pretrained_path, 
            map_location=lambda storage, loc: storage.cuda(device))
    else:
        pretrained_dict = torch.load(
            pretrained_path, 
            map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
