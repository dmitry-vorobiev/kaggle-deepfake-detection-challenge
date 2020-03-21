python -u src/distributed_launch.py --nproc_per_node=2 src/train.py \
  distributed.backend=nccl lr_scheduler.params.max_lr=0.002