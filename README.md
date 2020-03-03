# kaggle-deepfake-detection-challenge
My solution for the Deepfake Detection Challenge

## Installation
Assuming the user has a standart Anaconda3 environment
1. Install NVIDIA DALI. Instructions can be found [here](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html).

2. Create folder called `vendors`. It will be used for 3rd party code which lacks proper install mechanisms

3. Install Pytorch_Retinaface:
```
cd vendors
git clone https://github.com/biubug6/Pytorch_Retinaface.git
```
then go to `src/prepare_data.py` and change this line in the import section:
```python
sys.path.insert(0, '/home/dmitry/projects/dfdc/vendors/Pytorch_Retinaface')
```

4. Install other non-standart pip packages:
```
pip install h5py
```
