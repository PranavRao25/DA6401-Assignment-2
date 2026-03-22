import torch
import numpy as np
import keras
import yaml
from typing import Any
from dropout import Dropout
from pathlib import Path
from torch import nn

"""
INPUT: 256 * 256 * 3
LAYER 1 : 256 * 256 * 64
MAXPOOL : 128 * 128 * 64
LAYER 2 : 128 * 128 * 128
MAXPOOL : 64 * 64 * 128
LAYER 3 : 64 * 64 * 256
LAYER 4 : 64 * 64 * 256
MAXPOOL : 32 * 32 * 256
LAYER 5 : 32 * 32 * 512
LAYER 6 : 32 * 32 * 512
MAXPOOL : 16 * 16 * 512
LAYER 7 : 16 * 16 * 512
LAYER 8 : 16 * 16 * 512
MAXPOOL : 8  * 8  * 512
"""

class LazyVGG11Encode(nn.Module):
    """
        VGG11 CONVOLUTIONAL ENCODER SKELETON
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super(LazyVGG11Encode, self).__init__()
        self.p = 0.5

        self.conv1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1),  # 0
            nn.BatchNorm2d(num_features=64),  # 1
            nn.ReLU()
        )

        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 3
        )
        
        self.conv2 = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3, padding=1),  # 4
            nn.BatchNorm2d(128),  # 5
            nn.ReLU()
        )

        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7
        )

        self.conv3 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1),  # 8
            nn.BatchNorm2d(256),  # 9
            nn.ReLU(),  # 10
        )

        self.conv4 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1),  # 11
            nn.BatchNorm2d(256),  # 12
            nn.ReLU(),  # 13
        )

        self.maxpool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14
        )

        self.conv5 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),  # 15
            nn.BatchNorm2d(512),  # 16
            nn.ReLU(),  # 17
        )

        self.conv6 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),  # 18
            nn.BatchNorm2d(512),  # 19
            nn.ReLU(),  # 20
        )

        self.maxpool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 21
        )

        self.conv7 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),  # 22
            nn.BatchNorm2d(512),  # 23
            nn.ReLU(),  # 24
        )

        self.conv8 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),  # 25
            nn.BatchNorm2d(512),  # 26
            nn.ReLU(),  # 27
        )

        self.maxpool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28
        )

        self.conv_layers = [
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3,
            self.conv4,
            self.maxpool3,
            self.conv5,
            self.conv6,
            self.maxpool4,
            self.conv7,
            self.conv8,
            self.maxpool5
        ]
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
