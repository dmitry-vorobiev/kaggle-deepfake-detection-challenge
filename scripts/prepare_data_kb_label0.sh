export MKL_DEBUG_CPU_TYPE=5

python src/prepare_data.py --start 0 --end 30 --chunks $1 --label 0 \
 --num_frames 100 --stride 3 --num_pass 4 \
 --batch_size 64 --gpus 0 \
 --img_format webp --num_workers 4 --task_queue_depth 12 \
 --data_dir data/dfdc-videos \
 --save_dir data/dfdc-crops/webp \
 --det_weights data/weights/mobilenet0.25_Final.pth
