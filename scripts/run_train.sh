export OMP_NUM_THREADS=1

python -u src/distributed_launch.py --nproc_per_node=2 src/train.py \
  distributed.backend=nccl \
  lr_scheduler.params.max_lr=0.001 \
  data.train.loader.batch_size=15 \
  optimizer.step_interval=2 \
  train.epoch_length=8000 \
  train.checkpoints.load=/media/dmitry/data/outputs/2020-03-30/20-35-15/checkpoint_24000.pth