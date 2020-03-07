export MKL_DEBUG_CPU_TYPE=5

python src/prepare_data.py --start 0 --end 100 --chunks $1 --label 1 \
 --num_frames 30 --stride 10 --num_pass 1 \
 --batch_size 64 --gpus 0 \
 --img_format webp --num_workers 4 --task_queue_depth 12 \
 --data_dir data/dfdc-videos \
 --save_dir data/dfdc-crops/webp \
 --det_weights data/weights/mobilenet0.25_Final.pth