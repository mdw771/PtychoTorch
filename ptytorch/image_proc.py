from typing import Tuple, Literal

import torch
from torch import Tensor


def extract_patches_fourier_shift(
        image: Tensor, 
        positions: Tensor, 
        shape: Tuple[int, int]
    ) -> Tensor:
    """Extract patches from 2D object.

    :param image: the whole image.
    :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
    :param shape: a tuple giving the patch shape in pixels.
    """    
    # Floating point ranges over which interpolations should be done
    sys_float = (positions[:, 0] - (shape[0] - 1.0) / 2.0)
    sxs_float = (positions[:, 1] - (shape[1] - 1.0) / 2.0)
    
    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - 1
    eys = sys + shape[0] + 2
    sxs = sxs_float.floor().int() - 1
    exs = sxs + shape[1] + 2
    
    fractional_shifts = torch.stack([sys_float - sys - 1.0, sxs_float - sxs - 1.0], -1)
    
    pad_lengths = [max(-sxs.min(), 0), max(exs.max() - image.shape[1], 0), 
                   max(-sys.min(), 0), max(eys.max() - image.shape[0], 0)]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]
    
    patches = []
    for sy, ey, sx, ex in zip(sys, eys, sxs, exs):
        p = image[sy:ey, sx:ex]
        patches.append(p)
    patches = torch.stack(patches)
            
    # Apply Fourier shift to account for fractional shifts
    patches = fourier_shift(patches, -fractional_shifts)
    patches = patches[:, 1:-1, 1:-1]
    return patches


def place_patches_fourier_shift(
        image: Tensor, 
        positions: Tensor, 
        patches: Tensor,
        op: Literal['add', 'set'] = 'add'
    ) -> Tensor:
    """Place patches into a 2D object.
    
    :param image: the whole image.
    :param positions: a tensor of shape (N, 2) giving the center positions of the patches in pixels.
    :param patches: (N, H, W) tensor ofimage patches.
    """
    shape = patches.shape[-2:]
    
    # Floating point ranges over which interpolations should be done. +1 to shrink
    # patches by 1 pixel
    sys_float = (positions[:, 0] - (shape[0] - 1.0) / 2.0) + 1
    sxs_float = (positions[:, 1] - (shape[1] - 1.0) / 2.0) + 1
    
    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int()
    eys = sys + shape[0] - 2
    sxs = sxs_float.floor().int()
    exs = sxs + shape[1] - 2
    
    fractional_shifts = torch.stack([sys_float - sys, sxs_float - sxs], -1)
    
    pad_lengths = [max(-sxs.min(), 0), max(exs.max() - image.shape[1], 0), 
                   max(-sys.min(), 0), max(eys.max() - image.shape[0], 0)]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]
    
    patches = fourier_shift(patches, fractional_shifts)
    patches = patches[:, 1:-1, 1:-1]
    
    for i in range(patches.shape[0]):
        if op == 'add':
            image[sys[i]:eys[i], sxs[i]:exs[i]] += patches[i]
        elif op == 'set':
            image[sys[i]:eys[i], sxs[i]:exs[i]] = patches[i]
        
    # Undo padding
    image = image[pad_lengths[2]:image.shape[0] - pad_lengths[3], 
                  pad_lengths[0]:image.shape[1] - pad_lengths[1]]
    return image
    

def fourier_shift(images: Tensor, shifts: Tensor) -> Tensor:
    """Apply Fourier shift to a batch of images.

    :param images: a [N, H, W] tensor of images.
    :param shifts: a [N, 2] tensor of shifts in pixels.
    :return: shifted images.
    """
    ft_images = torch.fft.fft2(images)
    freq_y, freq_x = torch.meshgrid(torch.fft.fftfreq(images.shape[-2]), torch.fft.fftfreq(images.shape[-1]), indexing='ij')
    freq_x = freq_x.to(ft_images.device)
    freq_y = freq_y.to(ft_images.device)
    freq_x = freq_x.repeat(images.shape[0], 1, 1)
    freq_y = freq_y.repeat(images.shape[0], 1, 1)
    mult = torch.exp(1j * -2 * torch.pi * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1)))
    ft_images = ft_images * mult
    shifted_images = torch.fft.ifft2(ft_images)
    if not images.dtype.is_complex:
        shifted_images = shifted_images.real
    return shifted_images
    

if __name__ == '__main__':
    a = torch.zeros([10, 10]).float()
    patches = torch.arange(6).reshape(1, 6).repeat(6, 1).reshape(1, 6, 6)
    pos = torch.tensor([[3., 3.]])
    print(patches)
    print(place_patches_fourier_shift(a, pos, patches))
