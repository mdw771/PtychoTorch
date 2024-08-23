from typing import Optional, Union, Tuple, Type

from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from numpy import ndarray

from .utils import to_tensor


class PtychographyDataset(Dataset):
    
    def __init__(self, 
                 patterns: Union[Tensor, ndarray], 
                 positions: Union[Tensor, ndarray], 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patterns = to_tensor(patterns, device='cpu')
        self.positions = to_tensor(positions, device='cpu')
    
    def __getitem__(self, index):
        pattern = self.patterns[index]
        positions = self.positions[index]
        return positions, pattern
    
    def __len__(self):
        return len(self.patterns)