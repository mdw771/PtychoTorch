from typing import Union, Literal, Callable, Optional

import torch
from torch import Tensor
from torchvision.transforms import GaussianBlur
import numpy as np
from numpy import ndarray

from ptychotorch.propagation import propagate_far_field


default_complex_dtype = torch.complex64


def get_suggested_object_size(positions_px, probe_shape, extra=0):
    h = np.ceil(positions_px[:, 0].max() - positions_px[:, 0].min()) + probe_shape[0] + extra
    w = np.ceil(positions_px[:, 1].max() - positions_px[:, 1].min()) + probe_shape[1] + extra
    return (int(h), int(w))


def rescale_probe(
        probe: Union[ndarray, Tensor], 
        patterns: Union[ndarray, Tensor], 
        weights: Optional[Union[ndarray, Tensor]] = None
    ) -> None:
    """
    Scale probe so that the sum of intensity matches that of the diffraction patterns.

    :param probe: A (n_modes, h, w) or (n_opr_modes, n_modes, h, w) tensor of the probe.
    :param patterns: A (n, h, w) tensor of diffraction patterns. 
    :param weights: A (n_opr_modes,) tensor of weights for each OPR mode.
    :return: the scaled probe.
    """
    probe_tensor = torch.tensor(probe)
    
    if probe_tensor.ndim == 3:
        i_probe = (torch.abs(propagate_far_field(probe_tensor)) ** 2).sum().detach().cpu().numpy()
    else:
        weights = torch.tensor(weights)
        probe_corrected = (probe_tensor * weights[:, None, None, None]).sum(0)
        i_probe = (torch.abs(propagate_far_field(probe_corrected)) ** 2).sum().detach().cpu().numpy()
    
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


def add_additional_opr_probe_modes_to_probe(probe: Tensor, n_opr_modes_to_add: int) -> Tensor:
    assert probe.ndim == 4
    n_modes = probe.shape[1]
    opr_modes = torch.empty([n_opr_modes_to_add, n_modes, probe.shape[-2], probe.shape[-1]], dtype=get_default_complex_dtype())
    for i in range(n_opr_modes_to_add):
        for j in range(n_modes):
            mag = generate_gaussian_random_image(probe.shape[-2:], loc=0.01, sigma=0.01, smoothing=3.0).clamp(0.0, 1.0)
            phase = generate_gaussian_random_image(probe.shape[-2:], loc=0.0, sigma=0.05, smoothing=3.0).clamp(-torch.pi, torch.pi)
            opr_mode = mag * torch.exp(1j * phase)
            opr_modes[i, j, ...] = opr_mode
    probe = torch.cat([probe, opr_modes], dim=0)
    return probe


def generate_initial_opr_mode_weights(n_points: int, n_opr_modes: int, eigenmode_weight: float = 0.0) -> Tensor:
    """
    Generate initial weights for OPR modes, where the weights of the main OPR mode are set to 1,
    and the weights of eigenmodes are set to 0.

    :param n_points: number of scan points. 
    :param n_opr_modes: number of OPR modes.
    :param eigenmode_weight: initial weight for eigenmodes.
    :return: a (n_points, n_opr_modes) tensor of weights.
    """
    return torch.cat([torch.ones([n_points, 1]), torch.full([n_points, n_opr_modes - 1], eigenmode_weight)], dim=1)


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


def chunked_processing(func: Callable, common_kwargs: dict, chunkable_kwargs: dict, iterated_kwargs: dict, 
                       chunk_size: int = 96):
    """
    Breaks the data of a vectorized function into chunks and process chunks in sequence to
    reduce peak memory usage. 

    :param func: the callable to be executed.
    :param common_kwargs: a dictionary of arguments that should stay constant across chunks.
    :param chunkable_kwargs: a dictionary of arguments that should be chunked.
    :param iterated_kwargs: a dictionary of arguments that should be returned by `func`, then
        passed to `func` for the next chunk. The order of arguments should be the same as
        the returns of `func`.
    :return: the returns of `func` as if it is executed for the entire data.
    """
    full_batch_size = tuple(chunkable_kwargs.values())[0].shape[0]
    for key, value in tuple(chunkable_kwargs.items())[1:]:
        assert value.shape[0] == full_batch_size, \
            'All common arguments must have the same batch size, but {} has shape {}.'.format(key, value.shape)
    
    chunks_of_chunkable_args = []
    ind_st = 0
    while ind_st < full_batch_size:
        ind_end = min(ind_st + chunk_size, full_batch_size)
        chunk = {key: value[ind_st:ind_end] for key, value in chunkable_kwargs.items()}
        chunks_of_chunkable_args.append(chunk)
        ind_st = ind_end
    
    for kwargs_chunk in chunks_of_chunkable_args:
        ret = func(**common_kwargs, **kwargs_chunk, **iterated_kwargs)
        if isinstance(ret, tuple):
            for i, key in enumerate(iterated_kwargs.keys()):
                iterated_kwargs[key] = ret[i]
        else:
            iterated_kwargs[tuple(iterated_kwargs.keys())[0]] = ret
    if len(iterated_kwargs) == 1:
        return tuple(iterated_kwargs.values())[0]
    else:
        return tuple(iterated_kwargs.values())
    