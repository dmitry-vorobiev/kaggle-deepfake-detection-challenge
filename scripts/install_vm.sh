bucket=gs://tpu-pytorch/wheels
version=nightly+20200325

torch_wheel=torch-$version-cp36-cp36m-linux_x86_64.whl
torch_xla_wheel=torch_xla-$version-cp36-cp36m-linux_x86_64.whl
torchvision_wheel=torchvision-$version-cp36-cp36m-linux_x86_64.whl

for wheel in $torch_wheel $torch_xla_wheel $torchvision_wheel; do
  gsutil cp $bucket/$wheel .
  # python -m pip install $wheel
done

pip install --pre pytorch-ignite --no-deps

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/anaconda3/envs/xla/lib