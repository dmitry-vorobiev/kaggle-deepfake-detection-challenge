import numpy as np

import torch
from torchvision import ops

from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox


def detect(sample, model, cfg, device):
    num_frames, height, width, ch = sample.shape
    imgs, scale = prepare_imgs(sample, device)

    with torch.no_grad():
        loc, conf, landms = model(imgs)
    imgs, landms = None, None
    
    priorbox = PriorBox(cfg, image_size=(height, width))
    priors = priorbox.forward().to(device)

    dets = postproc_detections(loc, conf, priors, scale, cfg)
    return dets


def prepare_imgs(sample, device):
    n, h, w, c = sample.shape
    
    imgs = np.float32(sample)
    imgs -= (104, 117, 123)
    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = torch.from_numpy(imgs).to(device)

    scale = torch.tensor([w, h, w, h], device=device)
    return imgs, scale


def postproc_detections(
        locations, confidence, priors, 
        scale, cfg, resize=1):
    boxes = decode_batch(locations, priors, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = confidence.cpu().numpy()[:, :, 1]
    num_frames = scores.shape[0]
    dets = [postproc_frame(boxes[i], scores[i]) 
            for i in range(num_frames)]
    return dets


def decode_batch(loc, priors, variances):
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
        boxes, scores, 
        score_thresh=0.75, nms_thresh=0.4, 
        top_k=500, keep_top_k=5):
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


def postproc_frame_gpu(
        boxes, scores, 
        score_thresh=0.75, nms_thresh=0.4, 
        top_k=500, keep_top_k=5):
    inds = (scores > score_thresh).nonzero()[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    scores, idxs = scores.sort(descending=True)
    scores, idxs = scores[:top_k], idxs[:top_k]
    boxes = boxes[idxs]

    # do NMS
    keep = ops.nms(boxes, scores, nms_thresh)
    boxes = boxes[keep][:keep_top_k]
    scores = scores[keep][:keep_top_k]
    
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()[:, np.newaxis]
    dets = np.hstack([boxes, scores]).astype(np.float32, copy=False)
    return dets


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


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, 
            map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, 
            map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
