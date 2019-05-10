# EE225B Final Project

**Griffin and Hasith:Targetless Extrinsic Calibration of Camera and LiDAR 
using ML-based segmentation
**

download KITTI road data at `https://drive.google.com/file/d/15jBrlQlMJ51A1BcOSasFwXohaipZL7Kz/view?usp=sharing`
or http://www.cvlibs.net/datasets/kitti/eval_road.php

train.py : used for start training the model.
*.h5 : weights files after training.
logs and screenlogs : history of training terminal output.
demo.ipynb : test the model on testing images.
config.json : configurations and selections of architecture of the models, and training hyperparameters.
src / backend.py : where models are built.
      frontend.py : preparation of keras training.
      Datagen.py : data preprocessing and augmentation.

## Dependencies
      Opencv, imgaug, keras    They can be install using pip install 'names'.

## Usage
Image segmentation

    # Train a new model using kitti
    Usage: python3 train.py  --conf=./config.json

## Explaination 
   change the backend model in json file to train with different models. Currently support UNET,ENET,VGG16 from fast to slow and less accurate to most accurate. The keras architectures of the model are refered from the official website tutorials and several blogs.

## Results 
   Train the model yourself or see the logs of my training. The jupyter notebook demos examples of model results.
