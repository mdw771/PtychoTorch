import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

from ptytorch.data_structures import *
from ptytorch.io_handles import PtychographyDataset
from ptytorch.forward_models import Ptychography2DForwardModel
from ptytorch.utils import (get_suggested_object_size, set_default_complex_dtype, get_default_complex_dtype, 
                            rescale_probe)
from ptytorch.reconstructors import AutodiffReconstructor
from ptytorch.metrics import MSELossOfSqrt

from ptychodus.plugins.h5DiffractionFile import H5DiffractionFileReader
from ptychodus.plugins.h5ProductFile import H5ProductFileIO


torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)
set_default_complex_dtype(torch.complex64)

patterns = h5py.File('data/dp_250.hdf5', 'r')['dp'][...]

f_meta = h5py.File('data/metadata_250_truePos.hdf5', 'r')
probe = f_meta['probe'][...]
probe = rescale_probe(probe, patterns)
positions = np.stack([f_meta['probe_position_y_m'][...], f_meta['probe_position_x_m'][...]], axis=1)
pixel_size_m = 8e-9

positions_px = positions / pixel_size_m
dataset = PtychographyDataset(patterns, positions_px)
object = Object2D(
    data=torch.ones(get_suggested_object_size(positions_px, probe.shape[-2:]), dtype=get_default_complex_dtype()), 
    pixel_size_m=pixel_size_m,
    optimizable=True,
    optimizer_class=torch.optim.Adam,
    optimizer_params={'lr': 1e-4}
)
probe = Probe(
    data=probe,
    optimizable=True,
    optimizer_class=torch.optim.Adam,
    optimizer_params={'lr': 1e-4}
)
probe_positions = ProbePositions(
    data=positions_px,
    optimizable=False,
    optimizer_class=torch.optim.Adam,
    optimizer_params={'lr': 1e-3}
)

forward_model = Ptychography2DForwardModel(
    object=object,
    probe=probe,
    probe_positions=probe_positions
)

reconstructor = AutodiffReconstructor(
    dataset=dataset,
    forward_model=forward_model,
    batch_size=96,
    loss_function=MSELossOfSqrt(),
    n_epochs=128
)
reconstructor.build()
reconstructor.run()

recon = np.angle(reconstructor.forward_model.object.tensor.detach().cpu().numpy())
plt.imshow(recon)
plt.show()
