export OMP_NUM_THREADS=1

python -u src/distributed_launch.py --nproc_per_node=2 src/train.py \
  distributed.backend=nccl \
  model=efficient_kek loss=bce optimizer=lamb \
  lr_scheduler.params.max_lr=0.002 \
  data.train.loader.batch_size=64 \
  optimizer.step_interval=1 \
  train.epoch_length=4000 \
  data.train.sample.frames=5