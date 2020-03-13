# kaggle-deepfake-detection-challenge
My solution for the Deepfake Detection Challenge

## Installation
Assuming the user has a standard Anaconda3 environment
1. Install NVIDIA DALI. Instructions can be found [here](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html).

2. Create a new folder called `vendors` or something similar. It will be used for 3rd party code which lacks proper install mechanisms

3. Install [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface):
```bash
cd vendors
git clone https://github.com/biubug6/Pytorch_Retinaface.git
```
then go to `src/prepare_data.py` and update this line with path to Pytorch_Retinaface in your system:
```python
sys.path.insert(0, '/home/dmitry/projects/dfdc/vendors/Pytorch_Retinaface')
```

4. Download pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1).

5. Install other non-standart pip packages:
```
pip install h5py pytorch-ignite
```
