export MKL_DEBUG_CPU_TYPE=5

python src/prepare_data.py --start 0 --end 100 --chunks 2 --label 1 \
 --num_frames 30 --stride 10 --num_pass 1 \
 --score_thresh 0.75 --nms_thresh 0.4 --top_k 500 --keep_top_k 5 \
 --batch_size 64 --gpus 1,0 \
 --img_format webp --lossy --img_scale 2 --num_workers 3 --task_queue_depth 12 \
 --data_dir data/dfdc-videos \
 --save_dir /media/dmitry/other/dfdc-crops/webp_lossy \
 --det_weights data/weights/mobilenet0.25_Final.pth

# python src/prepare_data.py --start 0 --end 100 --chunks 2 --label 0 \
#  --num_frames 100 --stride 3 --num_pass 4 \
#  --score_thresh 0.75 --nms_thresh 0.4 --top_k 500 --keep_top_k 5 \
#  --batch_size 64 --gpus 1,0 \
#  --img_format webp --lossy --img_scale 2 --num_workers 3 --task_queue_depth 12 \
#  --data_dir data/dfdc-videos \
#  --save_dir /media/dmitry/other/dfdc-crops/webp_lossy \
#  --det_weights data/weights/mobilenet0.25_Final.pth
