import numpy as np
from torch import Tensor
from typing import Callable, List

from image import crop_square


def find_faces(frames: np.ndarray, detect_fn: Callable,
               max_face_num_thresh: float) -> List[np.ndarray]:
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
