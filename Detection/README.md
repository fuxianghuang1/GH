# Harmonic-detetion-loss
A PyTorch implementation of ['Reconcile Prediction Consistency for Balanced Object Detection'](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Reconcile_Prediction_Consistency_for_Balanced_Object_Detection_ICCV_2021_paper.pdf) on DSSD


### Preparation

#### Requirements: Python=3.5 and Pytorch=0.4.1

1. Install [Pytorch](http://pytorch.org/)

2. Download PASCAL VOC dataset
      
### Train and Test

1.Download the pretrained model [ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) and put it into ./weights

2.Change the dataset root path in ./data/config.py and some save dir path in ./train.py

3 Train the model
 ```Shell
 # train
 CUDA_VISIBLE_DEVICES=GPU_ID python train.py
 
 # Test model
 CUDA_VISIBLE_DEVICES=GPU_ID python test.py
 ```
 
