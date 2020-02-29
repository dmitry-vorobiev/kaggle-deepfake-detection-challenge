import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from typing import List, Callable

import cv2
import numpy as np
import pandas as pd
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import torch
from torch import Tensor

# TODO: make proper setup.py
sys.path.insert(0, '/home/dmitry/projects/dfdc/vendors/Pytorch_Retinaface')
from data import cfg_mnet

from dataset.utils import read_labels
from detectors.retinaface import detect, init_detector
from file_utils import mkdirs, dump_to_disk
from image import crop_square
from video import VideoPipe, read_frames_cv2


def get_file_list(df: pd.DataFrame, start: int, end: int, 
                  base_dir: str) -> List[str]:
    path_fn = lambda row: os.path.join(base_dir, row.dir, row.name)
    return df.iloc[start:end].apply(path_fn, axis=1).values.tolist()


def write_file_list(files: List[str], path: str) -> None:    
    with open(path, mode='w') as h:
        for i, f in enumerate(files):
            h.write(f'{f} {i}\n')


def find_faces(frames: np.ndarray, detect_fn: Callable, 
               num_face_round_thresh: int) -> None:
    detections = detect_fn(frames)
    if isinstance(frames, Tensor):
        frames = frames.cpu().numpy()
    num_faces = np.array(list(map(len, detections)), dtype=np.uint8)
    max_faces_per_frame = round_num_faces(num_faces, num_face_round_thresh)
    faces = []
    for f in range(len(frames)):
        for det in detections[f][:max_faces_per_frame]:
            face = crop_square(frames[f], det[:4])
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            if face is not None:
                faces.append(face)
    detections = None
    return faces


def round_num_faces(num_faces: int, frac_thresh: int) -> int:
    avg = num_faces.mean()
    fraction, integral = np.modf(avg)
    rounded = integral if fraction < frac_thresh else integral + 1
    return int(rounded)


def prepare_data(
        start=0, end: int=None, chunk_dirs: List[str]=None, 
        max_open_files=300, num_frames=30, stride=10, num_pass=1, gpu='0',
        batch_size=32, verbose=False, num_face_round_thresh=0.4,
        data_dir='', save_dir='', det_weights='', file_list_path='') -> None:
    df = read_labels(data_dir, chunk_dirs=chunk_dirs)
    if end is None:
        end = len(df)
    seq_len = num_frames // num_pass
    print('Max sequences per sample: %d' % num_pass)

    device = torch.device('cuda:{}'.format(gpu))
    cfg = {**cfg_mnet, 'batch_size': batch_size}
    detector = init_detector(cfg, det_weights, use_cpu=False).to(device)
    detect_fn = partial(detect, model=detector, cfg=cfg, device=device)
    
    for start_pos in range(start, end, max_open_files):
        end_pos = min(start_pos + max_open_files, end)
        files = get_file_list(df, start_pos, end_pos, data_dir)
        write_file_list(files, path=file_list_path)
        pipe = VideoPipe(file_list_path, seq_len=seq_len, stride=stride, device_id=int(gpu))
        pipe.build()
        num_samples = len(files)
        num_samples_read = pipe.epoch_size('reader')
        num_bad_samples = num_samples - num_samples_read / num_pass
        run_fallback_reader = num_bad_samples > 0
        if run_fallback_reader:
            proc_file_idxs = np.zeros(num_samples, dtype=np.bool)
            
        if num_samples_read > 0:
            data_iter = DALIGenericIterator(
                [pipe], ['frames', 'label'], num_samples_read, dynamic_shape=True)
            if verbose: 
                t0 = time.time()
            prev_idx = None
            for video_batch in data_iter:
                frames = video_batch[0]['frames'].squeeze(0)
                read_idx =  video_batch[0]['label'].item()
                abs_idx = start_pos + read_idx
                meta = df.iloc[abs_idx]
                faces = find_faces(frames, detect_fn, num_face_round_thresh)
                video_batch, frames = None, None
                
                dir_path = os.path.join(save_dir, meta.dir)
                append = prev_idx == read_idx
                dump_to_disk(faces, dir_path, meta.name[:-4], append=append)
                prev_idx = read_idx
                if run_fallback_reader:
                    proc_file_idxs[read_idx] = True
                if verbose:
                    t1 = time.time()
                    print('[%s][%6d][%.02f s] %s/%s' % (
                        str(device), abs_idx, t1 - t0, meta.dir, meta.name))
                    t0 = t1
        pipe, data_iter = None, None
        gc.collect()
        
        if run_fallback_reader:
            unproc_file_idxs = (~proc_file_idxs).nonzero()[0]
            num_bad_samples = len(unproc_file_idxs)
            if not num_bad_samples:
                continue
            print('Unable to parse %d videos with DALI' % num_bad_samples)
            print('Running fallback decoding through OpenCV...')
            for idx in unproc_file_idxs:
                if verbose: 
                    t0 = time.time()
                frames = read_frames_cv2(files[idx], num_frames)
                abs_idx = start_pos + idx
                meta = df.iloc[abs_idx]
                faces = find_faces(frames, detect_fn, num_face_round_thresh)
                dump_to_disk(faces, os.path.join(save_dir, meta.dir), meta.name[:-4])
                if verbose:
                    t1 = time.time()
                    print('[%s][%6d][%.02f s] %s/%s' % (
                        str(device), abs_idx, t1 - t0, meta.dir, meta.name))
    print('{}: DONE'.format(device))


def sizeof(data_dir: str, chunk_dirs: List[str]):
    df = read_labels(data_dir, chunk_dirs=chunk_dirs)
    return len(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--chunks', type=str, default='')
    parser.add_argument('--max_open_files', type=int, default=300, 
                        help='maximum open files to open with DALI pipe')
    parser.add_argument('--num_frames', type=int, default=30, 
                        help='max number of frames to use per each video')
    parser.add_argument('--stride', type=int, default=10, 
                        help='interval between consecutive frames')
    parser.add_argument('--num_pass', type=int, default=1, 
                        help='split parsing of each video into multiple '
                             'passes to save GPU memory')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size to use for detection')
    parser.add_argument('--data_dir', type=str, default='', 
                        help='where unpacked videos are stored')
    parser.add_argument('--save_dir', type=str, default='', 
                        help='where to save packed images')
    parser.add_argument('--det_weights', type=str, default='', 
                        help='weights for Pytorch_Retinaface model')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--num_face_round_thresh', type=float, default=0.4, 
                        help='round number of faces using custom threshold')


    args = parser.parse_args()
    verbose = not args.silent
    gpus = args.gpus.split(',')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    chunk_dirs = args.chunks.split(',')
    if len(chunk_dirs) > 0:
        chunk_dirs = [f'dfdc_train_part_{i}' for i in chunk_dirs]
        mkdirs(args.save_dir, chunk_dirs)
    else:
        chunk_dirs = None

    print('Reading from %s' % args.data_dir)
    print('Saving to %s' % args.save_dir)

    proc_fn = partial(prepare_data, 
        chunk_dirs=chunk_dirs, 
        max_open_files=args.max_open_files, 
        file_list_path='./temp.txt', 
        verbose=verbose,
        num_frames=args.num_frames,
        stride=args.stride, 
        num_pass=args.num_pass, 
        batch_size=args.batch_size,
        num_face_round_thresh=args.num_face_round_thresh,
        data_dir=args.data_dir, 
        save_dir=args.save_dir,
        det_weights=args.det_weights,
    )

    if len(gpus) > 1:
        if not args.end:
            args.end = sizeof(args.data_dir, chunk_dirs)
        num_samples_per_gpu = (args.end - args.start) // len(gpus)
        jobs = []
        with mp.Pool(len(gpus)) as proc_pool:
            for i, gpu in enumerate(gpus):
                job = proc_pool.apply_async(proc_fn, [], dict(
                    start=(num_samples_per_gpu * i), 
                    end=(num_samples_per_gpu * (i + 1)), 
                    gpu=gpu, 
                    file_list_path=f'./temp_{gpu}.txt'))
                jobs.append(job)
            for job in jobs:
                job.wait()
    else:
        proc_fn(start=args.start, 
                end=args.end, 
                gpu=gpus[0], 
                file_list_path='./temp.txt')
