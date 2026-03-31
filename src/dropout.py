from typing import Any
import torch.nn as nn
import torch

class Dropout(nn.Module):
    """
        Apply a mask on the input itself
        Make it zero
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if not(0 <= p <= 1):
            raise ValueError("probability has to be between 0 and 1")
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        mask = torch.bernoulli(torch.full_like(x, fill_value=self.p))
        input = x * mask
        return input

nn.Dropout()