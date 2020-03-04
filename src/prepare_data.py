import argparse
import concurrent.futures as futures
import gc
import os
import sys
import time
from functools import partial
from typing import Dict, List, Callable

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
            if os.path.isfile(f):
                h.write(f'{f} {i}\n')


def find_faces(frames: np.ndarray, detect_fn: Callable, 
               max_face_num_thresh: float) -> None:
    detections = detect_fn(frames)
    if isinstance(frames, Tensor):
        frames = frames.cpu().numpy()
    num_faces = np.array(list(map(len, detections)), dtype=np.uint8)
    max_faces = max_num_faces(num_faces, max_face_num_thresh)
    faces = []
    for f in range(len(frames)):
        for det in detections[f][:max_faces]:
            face = crop_square(frames[f], det[:4])
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            if face is not None:
                faces.append(face)
    detections = None
    return faces


def max_num_faces(num_faces_per_frame: np.ndarray, uniq_frac_thresh: float) -> int:
    uniq_vals, uniq_freq = np.unique(num_faces_per_frame, return_counts=True)
    mask = uniq_freq / len(num_faces_per_frame) > uniq_frac_thresh
    return uniq_vals[mask].max()


def wait_to_complete(tasks: List[futures.Future]) -> List:
    futures.wait(tasks)
    for task in tasks:
        _ = task.result()
    return []


def prepare_data(start: int, end: int, chunk_dirs: List[str]=None, gpu='0', 
                 args: Dict[str, any]=None) -> None:
    df = read_labels(args.data_dir, chunk_dirs=chunk_dirs)
    seq_len = args.num_frames // args.num_pass

    device = torch.device('cuda:{}'.format(gpu))
    cfg = {**cfg_mnet, 'batch_size': args.batch_size}
    detector = init_detector(cfg, args.det_weights, device).to(device)
    detect_fn = partial(detect, model=detector, cfg=cfg, device=device)
    file_list_path = '{}_{}'.format(args.file_list_path, gpu)

    with futures.ProcessPoolExecutor(args.num_workers) as subproc:
        tasks = []
        for start_pos in range(start, end, args.max_open_files):
            end_pos = min(start_pos + args.max_open_files, end)
            files = get_file_list(df, start_pos, end_pos, args.data_dir)
            write_file_list(files, path=file_list_path)
            pipe = VideoPipe(file_list_path, seq_len=seq_len, 
                             stride=args.stride, device_id=int(gpu))
            pipe.build()
            num_samples = len(files)
            num_samples_read = pipe.epoch_size('reader')
            num_bad_samples = num_samples - num_samples_read / args.num_pass
            run_fallback_reader = num_bad_samples > 0
            if run_fallback_reader:
                proc_file_idxs = np.zeros(num_samples, dtype=np.bool)
                
            if num_samples_read > 0:
                data_iter = DALIGenericIterator(
                    [pipe], ['frames', 'label'], 
                    num_samples_read, dynamic_shape=True)
                if args.verbose: 
                    t0 = time.time()
                prev_idx = None
                for video_batch in data_iter:
                    frames = video_batch[0]['frames'].squeeze(0)
                    read_idx =  video_batch[0]['label'].item()
                    faces = find_faces(frames, detect_fn, args.max_face_num_thresh)
                    video_batch, frames = None, None
                    abs_idx = start_pos + read_idx
                    meta = df.iloc[abs_idx]
                    dir_path = os.path.join(args.save_dir, meta.dir)
                    append = prev_idx == read_idx
                    task = subproc.submit(dump_to_disk, 
                        faces, dir_path, meta.name[:-4], 
                        args.img_format, append=append)
                    tasks.append(task)
                    prev_idx = read_idx
                    if run_fallback_reader:
                        proc_file_idxs[read_idx] = True
                    if args.verbose:
                        t1 = time.time()
                        print('[%s | DALI][%6d][%.02f s] %s/%s' % (
                            str(device), abs_idx, t1 - t0, meta.dir, meta.name))
                        t0 = t1
                    if len(tasks) > args.task_queue_depth * args.num_workers:
                        tasks = wait_to_complete(tasks)
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
                    if args.verbose: 
                        t0 = time.time()
                    frames = read_frames_cv2(files[idx], args.num_frames)
                    if frames is not None:
                        faces = find_faces(frames, detect_fn, args.max_face_num_thresh)
                        abs_idx = start_pos + idx
                        meta = df.iloc[abs_idx]
                        dir_path = os.path.join(args.save_dir, meta.dir)
                        task = subproc.submit(dump_to_disk, 
                            faces, dir_path, meta.name[:-4], 
                            args.img_format, append=False)
                        tasks.append(task)
                        if args.verbose:
                            t1 = time.time()
                            print('[%s | OpenCV][%6d][%.02f s] %s/%s' % (
                                str(device), abs_idx, t1 - t0, meta.dir, meta.name))
                    if len(tasks) > args.task_queue_depth * args.num_workers:
                        tasks = wait_to_complete(tasks)
    print('{}: DONE'.format(device))


def sizeof(data_dir: str, chunk_dirs: List[str]):
    df = read_labels(data_dir, chunk_dirs=chunk_dirs)
    return len(df)


def parse_args() -> Dict[str, any]:
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
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='num subproc per each GPU')
    parser.add_argument('--task_queue_depth', type=int, default=20, 
                        help='limit the amount of unfinished tasks per each worker')
    parser.add_argument('--max_face_num_thresh', type=float, default=0.25,
                        help='cut detections based on the frequency encoding')
    parser.add_argument('--img_format', type=str, default='png', 
                        choices=['png', 'webp'])
    args = parser.parse_args()
    args.verbose = not args.silent
    args.file_list_path = './temp'
    return args


if __name__ == '__main__':
    args = parse_args()
    gpus = args.gpus.split(',')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    chunk_dirs = args.chunks.split(',')
    if len(chunk_dirs) > 0:
        chunk_dirs = [f'dfdc_train_part_{i}' for i in chunk_dirs]
        mkdirs(args.save_dir, chunk_dirs)
    else:
        chunk_dirs = None

    print('reading from %s' % args.data_dir)
    print('saving to %s' % args.save_dir)
    print('DALI settings:\n'
          '  max simultaneosly open files: %d\n'
          '  num pass per video: %d' % (args.max_open_files, args.num_pass))

    proc_fn = partial(prepare_data, chunk_dirs=chunk_dirs, args=args)

    start, end = args.start, args.end
    if not end:
        end = sizeof(args.data_dir, chunk_dirs)

    if len(gpus) > 1:
        num_samples_per_gpu = (end - start) // len(gpus)
        jobs = []
        with futures.ProcessPoolExecutor(len(gpus)) as ex:
            for i, gpu in enumerate(gpus):
                job = ex.submit(proc_fn,
                    start=(start + num_samples_per_gpu * i), 
                    end=(start + num_samples_per_gpu * (i+1)), 
                    gpu=gpu)
                jobs.append(job)
            futures.wait(jobs)
        for job in jobs:
            _ = job.result()
    else:
        proc_fn(start, end, gpu=gpus[0])
