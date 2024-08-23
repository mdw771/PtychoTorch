import torch
from torch import Tensor


class MSELossOfSqrt(torch.nn.MSELoss):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        l = super().forward(torch.sqrt(y_pred), torch.sqrt(y_true))
        return l
