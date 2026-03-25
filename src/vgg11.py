import torch
import numpy as np
import keras
import yaml
from dropout import Dropout
from pathlib import Path
from torch import nn

class VGG11(nn.Module):
    """
        VGG11 was used for image recognition and classification

        VGG11 architecture with Batch normalization
        BATCH NORM BEFORE ACTIVATIONS
        DROPOUT AFTER ACTIVATIONS
        DROPOUT BEFORE FINAL OUTPUT LAYER
        DROPOUT AFTER MAXPOOL
        DROPOUT IN CNN SPARINGLY
    """

    def __init__(self, in_channels, num_classes=37):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.p = 0.5

        # convolutional layers
        self.conv_layers = nn.Sequential(  # ENCODING LAYER
            # BLOCK 1
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # MAXPOOL
            nn.MaxPool2d(kernel_size=2, stride=2),

            # BLOCK 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # MAXPOOL
            nn.MaxPool2d(kernel_size=2, stride=2),

            # BLOCK 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # BLOCK 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # MAXPOOL
            nn.MaxPool2d(kernel_size=2, stride=2),

            # BLOCK 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # BLOCK 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # MAXPOOL
            nn.MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.p), # DROPOUT

            # BLOCK 7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # BLOCK 8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # MAXPOOL
            nn.MaxPool2d(kernel_size=2, stride=2),
            Dropout(self.p),
        )
        # fully connected linear layers
        # Does it requires BatchNorm1D?
        self.linear_layers = nn.Sequential(  # classification layer
            # LINEAR 1
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 2
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 3 (OUTPUT LAYER)
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class LazyVGG11(nn.Module):
    """
        VGG11 was used for image recognition and classification

        VGG11 architecture with Batch normalization
        BATCH NORM BEFORE ACTIVATIONS
        DROPOUT AFTER ACTIVATIONS
        DROPOUT BEFORE FINAL OUTPUT LAYER
        DROPOUT AFTER MAXPOOL
        DROPOUT IN CNN SPARINGLY
    """

    def __init__(self, in_channels, num_classes=37):
        super(LazyVGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
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
        # fully connected linear layers
        # Does it requires BatchNorm1D?
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
