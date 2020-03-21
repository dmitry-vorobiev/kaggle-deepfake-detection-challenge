export CUDA_VISIBLE_DEVICES=0,1
export RANK=0
export WORLD_SIZE=-1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
# export NCCL_SOCKET_IFNAME=eth0

python src/train.py distributed.backend='nccl' distributed.local_rank=0 model=frodo
python src/train.py distributed.backend='nccl' distributed.local_rank=1 model=frodo