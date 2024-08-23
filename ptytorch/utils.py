from typing import Union

import torch
from torch import Tensor
import numpy as np
from numpy import ndarray


default_complex_dtype = torch.complex64


def get_suggested_object_size(positions_px, probe_shape):
    h = np.ceil(positions_px[:, 0].max() - positions_px[:, 0].min()) + probe_shape[0]
    w = np.ceil(positions_px[:, 1].max() - positions_px[:, 1].min()) + probe_shape[1]
    return (int(h), int(w))


def rescale_probe(probe: ndarray, patterns: ndarray):
    i_data = np.sum(np.mean(patterns, axis=0))
    i_probe = np.sum(np.array([np.abs(np.fft.fft2(p)) ** 2 for p in probe]))
    factor = i_data / i_probe
    probe = probe * np.sqrt(factor)
    return probe


def to_tensor(data: Union[ndarray, Tensor], device=None, dtype=None) -> Tensor:
    if isinstance(data, np.ndarray):
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


def set_default_complex_dtype(dtype):
    global default_complex_dtype
    default_complex_dtype = dtype
    

def get_default_complex_dtype():
    return default_complex_dtype

