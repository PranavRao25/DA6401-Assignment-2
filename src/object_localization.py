import torch
import torch.nn as nn
import yaml
from typing import Any
from pathlib import Path
from encoder import LazyVGG11Encode
from dropout import Dropout

"""
1. TRAIN THE VGG ENCODER FIRST
2. TRAIN THE REGRESSION DECODER USING L2 LOSS
3. AT TEST TIME, USE BOTH CLASSIFICATION AND REGRESSION HEADS
TODO: CHECK IF IT IS A PER-CLASS REGRESSION
"""

class ObjectLocalizer(nn.Module):
    """
        FINETUNE THE VGG ENCODER FIRST
        TO PERFORM SINGLE OBJECT LOCALIZATION TASK
        TAKE THE CONVOLUTIONAL ENCODER SKELETON OF VGG11
        ADD THE REGRESSION DECODER LAYERS FOR BOUNDING BOX PREDICTION
        LOSS FUNCTION : IoU
        TODO: FIND OUT THE  RANGE OF THE BOUNDING BOX ANNOTATIONS
    """

    def __init__(self, in_channels, num_classes = 37, output_size=4, p=0.5, *args: Any, **kwargs: Any) -> None:
        super(ObjectLocalizer, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.output_size = 4
        self.num_classes = num_classes
        self.p = p

        self.encoder = LazyVGG11Encode()
        self.regression_decoder = nn.Sequential(
            # LINEAR 1
            nn.Flatten(),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 2
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 3
            nn.Linear(in_features=4096, out_features=self.output_size)
        )

        self.classification_layer = nn.Sequential(
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

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        bounding_boxes = self.regression_decoder(x)
        object_class = self.classification_layer(x)
        return bounding_boxes, object_class
    