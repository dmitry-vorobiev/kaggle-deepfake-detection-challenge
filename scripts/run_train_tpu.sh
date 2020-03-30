python src/train.py tpu.enabled=true tpu.num_cores=8 \
  lr_scheduler.params.max_lr=0.002 data.train.loader.batch_size=16 \
  optimizer.step_interval=1 train.epoch_length=8000 \
  data.train.loader.workers=1 data.val.loader.workers=1 \
  logging.iter_freq=50
