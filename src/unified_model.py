import torch
import torch.nn as nn
import yaml
from typing import Any
from pathlib import Path
from encoder import LazyVGG11Encode
from semantic_seg import LazyVGG11Upsample
from heads import *
import torch.nn.functional as F
from dropout import Dropout

class UnifiedVGG11(nn.Module):
    def __init__(self, num_classes = 37, output_size=4, p=0.5, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.regr_output_size = output_size
        self.p = p

        self.encoder = LazyVGG11Encode()
        encoder_output_dim = 512*7*7
        
        self.regression_head = RegressionVGG11(encoder_output_dim, self.regr_output_size, p)  # TASK 2
        
        self.classification_head = ClassificationVGG11(encoder_output_dim, self.num_classes, p)  # TASK 1

        # TASK 3
        self.semantic_intermediate_layer = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.upsample = LazyVGG11Upsample()
        self.semantic_head = nn.Sequential(
            nn.LazyConv2d(num_classes, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        # PASS THROUGH ENCODER LAYER FIRST
        s1 = self.encoder.conv1(x)
        x = self.encoder.maxpool1(s1)

        s2 = self.encoder.conv2(x)
        x = self.encoder.maxpool2(x)

        x = self.encoder.conv3(x)
        s4 = self.encoder.conv4(x)
        x = self.encoder.maxpool3(s4)

        x = self.encoder.conv5(x)
        s6 = self.encoder.conv6(x)
        x = self.encoder.maxpool4(s6)

        x = self.encoder.conv7(x)
        s8 = self.encoder.conv8(x)
        x = self.encoder.maxpool5(x)

        # CLASSIFICATION
        object_class = self.classification_head(x)

        # REGRESSION
        bounding_boxes = self.regression_head(x)

        # SEMANTIC SEGMENTATION
        x = self.semantic_intermediate_layer(x)

        x = self.upsample.deconv1(x)
        x = self._pad_concat(s8, x)
        x = self.upsample.conv1(x)
        x = self.upsample.conv2(x)

        x = self.upsample.deconv2(x)
        x = self._pad_concat(s6, x)
        x = self.upsample.conv3(x)
        x = self.upsample.conv4(x)

        x = self.upsample.deconv3(x)
        x = self._pad_concat(s4, x)
        x = self.upsample.conv5(x)
        x = self.upsample.conv6(x)

        x = self.upsample.deconv4(x)
        x = self._pad_concat(s2, x)
        x = self.upsample.conv7(x)

        x = self.upsample.deconv5(x)
        x = self._pad_concat(s1, x)
        x = self.upsample.conv8(x)

        x = self.semantic_head(x)
        
        return object_class, bounding_boxes, x
    
    def _pad_concat(self, skip: torch.Tensor, upsampled: torch.Tensor):
        diffY = skip.size()[2] - upsampled.size()[2]
        diffX = skip.size()[3] - upsampled.size()[3]

        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX //2,
                                              diffX // 2, diffX - diffX //2])
        
        return torch.cat([skip, upsampled], dim=1)
