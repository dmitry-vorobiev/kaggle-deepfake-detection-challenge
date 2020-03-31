export OMP_NUM_THREADS=1

python -u src/distributed_launch.py --nproc_per_node=2 src/train.py \
  distributed.backend=nccl \
  model=efficient_kek data=effnet_b3 \
  loss=smooth_bce optimizer=lamb \
  lr_scheduler.params.max_lr=0.002 \
  data.train.loader.batch_size=64 \
  data.val.loader.batch_size=64 \
  optimizer.step_interval=1 \
  train.epoch_length=2000 \
  train.checkpoints.interval_iteration=200 \
  data.train.sample.frames=10