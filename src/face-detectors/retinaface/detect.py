import numpy as np

from torchvision import ops


def postproc_frame_gpu(
        boxes, scores, 
        score_thresh=0.75, nms_thresh=0.4, 
        top_k=500, keep_top_k=50):
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