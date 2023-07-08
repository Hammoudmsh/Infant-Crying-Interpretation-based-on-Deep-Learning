import os
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
# install pydub for using HighPassFilter and play
# import simpleaudio as sa
import matplotlib.pyplot as plt
#from helper import _plot_signal_and_augmented_signal
from IPython.display import Audio
import librosa.display as dsp
# import mir_eval
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary 
import os


from torch import nn
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor

import torchvision

i =0
class FeedForwardNet(nn.Module):

    def __init__(self, inpNum, shape, classesNum):
        super().__init__()
        print("***************", shape)
        #nn.Dropout2d(0.1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inpNum, 16, 3, 1,2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Dropout(0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1,2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1,2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1,2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Dropout(0.1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 2, 1,2), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 2, 1,2), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, 2, 1,2), nn.BatchNorm2d(1024), nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        linear_size = self.get_size(shape)
        self.linear1 = nn.Linear(linear_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, classesNum)

        # self.softmax = nn.Softmax(dim=1)
        
                
    def forward(self, inp):
        global i
        
        



        x = self.conv1(inp)



        x = self.conv2(x)



        x = self.conv3(x)



        x = self.conv4(x)

        
        
        x = self.conv5(x)

          
          
        x = self.conv6(x)

          
        x = self.conv7(x)



        x = x.view(-1, int(x.numel()/x.shape[0]))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = self.softmax(x)
        
        return x 
    
    def get_size(self, shape):
        output_shape = torch.zeros(shape)
        output_shape = self.conv1(output_shape)
        output_shape = self.conv2(output_shape)
        output_shape = self.conv3(output_shape)
        output_shape = self.conv4(output_shape)
        output_shape = self.conv5(output_shape)
        output_shape = self.conv6(output_shape)
        output_shape = self.conv7(output_shape)

#         Return batch_size (here, 1) * channels * height * width of input_sample
        return output_shape.numel();
