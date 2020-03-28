export OMP_NUM_THREADS=1

python -u src/distributed_launch.py --nproc_per_node=2 src/train.py \
  distributed.backend=nccl \
  lr_scheduler.params.max_lr=0.001 \
  data.train.loader.batch_size=16 \
  optimizer.step_interval=2 \
  train.epoch_length=15000 \
  train.checkpoints.load=/media/dmitry/data/outputs/2020-03-27/15-07-23/checkpoint_45000.pth