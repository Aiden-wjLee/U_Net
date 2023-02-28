# U-Net
U_Net using Docker.  
U-Net is famous Fully Convolutional Networks (FCN).  
In the U-Net thesis, they apply to biomedical image annotation.  
I need semantic segmentation Network operating in gray-scale images.  
So I use and test U-Net.

## Reference  
I use [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28#auth-Thomas-Brox)

## Setup environment  
Code has been tested with pytorch 1.12.1 and NVIDIA. 

## Train and Visualization
this project utilize Dockerfile that contains images for training and visualization.   

To run Train,  
Put json file in this directory path (./trainval_train.json) or (./trainval_test.json)   
and Put images (./images_train/*.png) or (./images_test/*.png)  
I referenced [labelme](https://github.com/wkentaro/labelme) and [labelme2coco](https://github.com/fcakyon/labelme2coco)

```
docker-compose up   
```
```
docker-compose run unet python [options]
```
- options:  
  - train.py   
  - mIoU_show.py  
  - mIoU.py
  

```
docker pull nvcr.io/nvidia/pytorch:22.04-py3
```
(if you fall into error when you make images, you can use here first.)
