import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import torchvision

from matplotlib.patches import Rectangle
import matplotlib
from pycocotools.coco import COCO

import json
import cv2
import random
import skimage.io as io
#from transform import transform_COCO

#from utils import *
#from utils import transform

import string
import matplotlib.gridspec as gridspec
import pandas as pd
#%matplotlib inline
import time 
import random

class UNet(nn.Module): ##nn.Module 상속
    """
    U_Net 모델을 만든다.
        functions:
            CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)함수
        
    """
    def __init__(self): 
        super(UNet, self).__init__()
        """
        CBR2d함수를 반복적으로 사용하여 Unet 생성
            Args:
                self.enc'x'_'y': CBR을 각 'x'층에서 2번씩 진행하기 위한 변수 (채널 증가)
                self.pool'z' : CBR2d 두번 시행 후 Maxpool2d진행 (크기 감소) 

                self.dec'x'_'y' : CBR을 각 'x'층에서 2번씩 진행하기 위한 변수 (채널 감소)
                self.unpool'z' : CBR2d 두번 시행 후 ConvTranspose2d진행 (크기 증가)
        """
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            """
            이후 반복적으로 사용될 Conv, BatchNorm, ReLU를 정의
                Args : 
                    cbr: nn.Sequention로 위 3개 정의 후 return
                Returns: 
                    cbr
            """
            cbr=nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )
            return cbr
            
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
    

  
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self,x):
        """
        UNet class의 forward
            Args:
                enc'x'_'y' : x층에서 2번 CBR2d 진행 (채널 증가)
                pool'z' : z층에서 maxpool2d 진행 (크기 감소)

                dec'x'_'y' : x층에서 2번 CBR2d 진행 (채널 감소)
                unpool'z' : x층 이후 z층으로 Convtranspose2d 진행 (크기 증가)
                cat'z' : z층에서 이전 계산한 결과를 torch.cat을 통해 더한다. (residual 방식)

            return:
                x : dec1_1 이후 채널을 1로 만들어주기 위해 self.fc 적용 후 리턴
        """
        enc1_1 = self.enc1_1(x)
        
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        #-----------------------#

        dec5_1 = self.dec5_1(enc5_1)
        unpool4=self.unpool4(dec5_1)

        cat4=torch.cat((unpool4, enc4_2),dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3=self.unpool3(dec4_1)

        ############print("unpool3.size: ",unpool3.shape)
        cat3=torch.cat((unpool3, enc3_2),dim=1)
        ############print("unpool3.size: ",cat3.shape)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2=self.unpool2(dec3_1)
        
        cat2=torch.cat((unpool2, enc2_2),dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1=self.unpool1(dec2_1)

        cat1=torch.cat((unpool1, enc1_2),dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x=self.fc(dec1_1)

        return x
