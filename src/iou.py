from typing import Any
import torchvision
import torch
import torch.nn as nn

class IoU(nn.Module):
    """
        Calculates the IoU loss as :
        L = 1 - I/U
        The bounding box is given by [x1, y1, x2, y2]

        x1 < x2
        x_1 < x_2
        y1 > y2
        y_1 > y_2

        get C = (min(x1, x_1), max(y1, y_1), max(x2, x_2), min(y2, y_2))

        I = (max(x1, x_1), min(y1, y_1), min(x2, x_2), max(y2, y_2))

        A area = |x1 - x2| * |y1 - y2|
        B area = |x_1 - x_2| * |y_1 - y_2|

        U = A area + B area - I
    """

    def __init__(self, fmt: str = "xyxy", smooth = 1.0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.smooth = smooth
    
    def forward(self, predict_box: torch.Tensor, true_box: torch.Tensor):
        x1, y1, x2, y2 = predict_box
        x_1, y_1, x_2, y_2 = true_box

        x1_inter, y1_inter, x2_inter, y2_inter = (max(x1, x_1), min(y1, y_1), min(x2, x_2), max(y2, y_2))
        I = torch.abs(x1_inter - x2_inter) * torch.abs(y1_inter - y2_inter) + self.smooth

        area1 = torch.abs(x1 - x2) * torch.abs(y1 - y2)
        area2 = torch.abs(x_1 - x_2) * torch.abs(y_1 - y_2)
        U = area1 + area2 - I + self.smooth

        return 1 - I/U
