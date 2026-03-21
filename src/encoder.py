import torch
import numpy as np
import keras
import yaml
from typing import Any
from dropout import Dropout
from pathlib import Path
from torch import nn

class LazyVGG11Encode(nn.Module):
    """
        VGG11 CONVOLUTIONAL ENCODER SKELETON
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super(LazyVGG11Encode, self).__init__()
        self.p = 0.5

        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.p), # DROPOUT
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.p),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x
