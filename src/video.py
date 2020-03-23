import os
import cv2
import numpy as np
import nvidia.dali as dali
from typing import List, Optional, Tuple


class VideoPipe(dali.pipeline.Pipeline):
    def __init__(self, file_list: str, seq_len: int, stride: int, 
                 batch_size=1, num_threads=1, device_id=0):
        super(VideoPipe, self).__init__(
            batch_size, num_threads, device_id, seed=3)
        
        self.input = dali.ops.VideoReader(
            device='gpu', file_list=file_list, sequence_length=seq_len, 
            stride=stride, shard_id=0, num_shards=1)

    def define_graph(self):
        output, labels = self.input(name='reader')
        return output, labels


def read_frames_cv2(path: str, num_frames: int, jitter=0, seed=None) -> Optional[np.ndarray]:
    """Reads frames that are always evenly spaced throughout the video.

    Arguments:
        path: the video file
        num_frames: how many frames to read, -1 means the entire video
            (warning: this will take up a lot of memory!)
        jitter: if not 0, adds small random offsets to the frame indices;
            this is useful so we don't always land on even or odd frames
        seed: random seed for jittering; if you set this to a fixed value,
            you probably want to set it only on the first video 

    Original: https://www.kaggle.com/humananalog/deepfakes-inference-demo
    """
    if not os.path.isfile(path):
        print('No such file: %s' % path)
        return None
    capture = cv2.VideoCapture(path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0: 
        return None
    frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
    if jitter > 0:
        np.random.seed(seed)
        jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
        frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)
    result = read_frames_at_indices(path, capture, frame_idxs)
    capture.release()
    return result


def read_frames_at_indices(path: str, capture: cv2.VideoCapture,
                           frame_idxs: np.ndarray) -> Optional[np.ndarray]:
    try:
        frames = []
        next_idx = 0
        for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
            ret = capture.grab()
            if not ret:
                print('Unable to grab frame %d from %s' % (frame_idx, path))
                break
            if frame_idx == frame_idxs[next_idx]:
                ret, frame = capture.retrieve()
                if not ret or frame is None:
                    print('Unable to retrieve frame %d from %s' % (frame_idx, path))
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                next_idx += 1
        if len(frames) > 0:
            return np.stack(frames)
        else:
            print('No frames have been read from %s' % path)
            return None
    except Exception as e:
        print('Unable to read %s' % path)
        print(e)
        return None


def parse_meta(files: List[str]) -> np.ndarray:
    meta = np.zeros((len(files), 3))
    for i, path in enumerate(files):
        cap = cv2.VideoCapture(path)
        meta[i, 0] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        meta[i, 1] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        meta[i, 2] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap.release()
    return meta


def split_files_by_res(files_meta: np.ndarray, min_freq: int
                       ) -> Tuple[List[np.ndarray], np.ndarray]:
    px_count = files_meta[:, 1] * files_meta[:, 2]
    clusters, freq = np.unique(px_count, return_counts=True)
    split_masks = [(px_count == c) for i, c in enumerate(clusters)
                   if freq[i] >= min_freq]
    size_factor = clusters / (1920 * 1080)
    return split_masks, size_factor

