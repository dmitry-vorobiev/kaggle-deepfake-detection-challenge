export MKL_DEBUG_CPU_TYPE=5

python src/prepare_data.py --start 0 --end 100 --chunks 48 \
 --num_frames 30 --stride 10 --num_pass 1 \
 --batch_size 64 --gpus 1,0 \
 --img_format png --num_workers 1 \
 --data_dir data/dfdc-videos \
 --save_dir data/dfdc-crops/hdf5 \
 --det_weights data/weights/mobilenet0.25_Final.pth
