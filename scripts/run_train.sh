export OMP_NUM_THREADS=1

python -u src/distributed_launch.py --nproc_per_node=2 src/train.py \
  distributed.backend=nccl \
  lr_scheduler.params.max_lr=0.004 optimizer.step_interval=2