export MKL_DEBUG_CPU_TYPE=5

python src/prepare_data.py --start 0  --chunks $1 --label 0 \
 --num_frames 32 --stride 1 --num_pass 4 \
 --batch_size 64 --gpus 0 \
 --img_format webp --num_workers 4 --task_queue_depth 12 \
 --data_dir data/dfdc-videos \
 --save_dir data/dfdc-crops/webp_seq \
 --det_weights data/weights/mobilenet0.25_Final.pth
