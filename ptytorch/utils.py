from typing import Union, Literal

import torch
from torch import Tensor
from torchvision.transforms import GaussianBlur
import numpy as np
from numpy import ndarray

from ptytorch.propagation import propagate_far_field


default_complex_dtype = torch.complex64


def get_suggested_object_size(positions_px, probe_shape, extra=0):
    h = np.ceil(positions_px[:, 0].max() - positions_px[:, 0].min()) + probe_shape[0] + extra
    w = np.ceil(positions_px[:, 1].max() - positions_px[:, 1].min()) + probe_shape[1] + extra
    return (int(h), int(w))


def rescale_probe(probe: Union[ndarray, Tensor], patterns: Union[ndarray, Tensor]):
    probe_tensor = torch.tensor(probe)
    i_probe = (torch.abs(propagate_far_field(probe_tensor)) ** 2).sum().detach().cpu().numpy()
    patterns = to_numpy(patterns)
    i_data = np.sum(np.mean(patterns, axis=0))
    factor = i_data / i_probe
    probe = probe * np.sqrt(factor)
    return probe


def generate_initial_object(shape: tuple[int, ...], method: Literal['random'] = 'random') -> Tensor:
    if method == 'random':
        obj_mag = generate_gaussian_random_image(shape, loc=0.9, sigma=0.1, smoothing=3.0)
        obj_mag = obj_mag.clamp(0.0, 1.0)
        obj_phase = generate_gaussian_random_image(shape, loc=0.0, sigma=0.5, smoothing=3.0)
        obj_phase = obj_phase.clamp(-torch.pi, torch.pi)
        obj = obj_mag * torch.exp(1j * obj_phase)
    else:
        raise ValueError(f'Unknown object initialization method: {method}')
    obj = obj.type(get_default_complex_dtype())
    return obj


def generate_gaussian_random_image(shape: tuple[int, ...], loc: float = 0.9, sigma: float = 0.1, 
                                   smoothing: float = 3.0) -> Tensor:
    img = torch.randn(shape, dtype=torch.get_default_dtype()) * sigma + loc
    if smoothing > 0.0:
        img = GaussianBlur(kernel_size=(9, 9), sigma=(3, 3))(img[None, None, :, :])
        img = img[0, 0, ...]
    return img
    


def to_tensor(data: Union[ndarray, Tensor], device=None, dtype=None) -> Tensor:
    if isinstance(data, (np.ndarray, list, tuple)):
        data = torch.tensor(data)
            
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        if data.dtype.is_complex:
            dtype = get_default_complex_dtype()
        elif not data.dtype.is_complex:
            dtype = torch.get_default_dtype()
    
    if data.dtype != dtype:
        data = data.type(dtype)
    if str(data.device) != str(device):
        data = data.to(device)
    return data


def to_numpy(data: Union[ndarray, Tensor]) -> ndarray:
    if isinstance(data, Tensor):
        data = data.detach().cpu().numpy()
    return data


def set_default_complex_dtype(dtype):
    global default_complex_dtype
    default_complex_dtype = dtype
    

def get_default_complex_dtype():
    return default_complex_dtype

