# export MKL_NUM_THREADS=1
export MKL_DEBUG_CPU_TYPE=5

python src/prepare_data.py --start 0 --chunks 1,2,3,4,5,6 \
 --num_frames 30 --stride 10 --num_pass 1 \
 --batch_size 32 --gpus 0,1 \
 --data_dir data/dfdc-videos \
 --save_dir data/dfdc-crops/hdf5 \
 --det_weights data/weights/mobilenet0.25_Final.pth
