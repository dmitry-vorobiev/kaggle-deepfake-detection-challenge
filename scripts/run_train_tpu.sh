python src/train.py tpu.enabled=true tpu.num_cores=1 \
  lr_scheduler.params.max_lr=0.001 data.train.loader.batch_size=16 \
  optimizer.step_interval=1 train.epoch_length=16000
