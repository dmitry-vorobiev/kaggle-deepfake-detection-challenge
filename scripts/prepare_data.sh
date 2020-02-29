python src/prepare_data.py --start 0 --end 100 --chunks 2 \
 --num_frames 30 --stride 10 --num_pass 1 \
 --batch_size 32 --gpus 1,0 \
 --data_dir data/dfdc-videos \
 --save_dir /media/dmitry/data/hdf5 \
 --det_weights data/weights/mobilenet0.25_Final.pth
