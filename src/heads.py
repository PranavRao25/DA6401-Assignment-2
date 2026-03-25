import torch
import numpy as np
import keras
import yaml
from typing import Any
from dropout import Dropout
from pathlib import Path
from torch import nn

class RegressionVGG11(nn.Module):
    def __init__(self, input_size, output_size, p, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.input_size = input_size
        self.output_size = output_size

        self.head = nn.Sequential(
            # LINEAR 1
            nn.Flatten(),
            nn.Linear(in_features=self.input_size, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 2
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 3
            nn.Linear(in_features=4096, out_features=self.output_size)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.head(x)
        return x

class ClassificationVGG11(nn.Module):
    def __init__(self, input_size, output_size, p, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.p = p

        self.head = nn.Sequential(
            # LINEAR 1
            nn.Linear(in_features=self.input_size, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 2
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            Dropout(self.p),

            # LINEAR 3 (OUTPUT LAYER)
            nn.Linear(in_features=4096, out_features=self.output_size)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.head(x)
        return x
