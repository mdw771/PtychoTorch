import torch
from torch import Tensor
from torch.nn import ModuleList

from ptytorch.data_structures import Variable, Object2D, Probe, ProbePositions


class ForwardModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.optimizable_variables: ModuleList[Variable] = ModuleList()
        
    def register_optimizable_parameters(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    
class Ptychography2DForwardModel(ForwardModel):
    
    def __init__(
            self, 
            object: Object2D, 
            probe: Probe, 
            probe_positions: ProbePositions, 
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.object = object
        self.probe = probe
        self.probe_positions = probe_positions
        
        self.register_optimizable_parameters()
        
    def register_optimizable_parameters(self):
        for var in [self.object, self.probe, self.probe_positions]:
            if var.optimizable:
                # self.register_parameter(name=var.name, param=torch.nn.Parameter(var.tensor))
                self.optimizable_variables.append(var)

    def forward(self, positions: Tensor, return_object_patches: bool = False) -> Tensor:
        """Run ptychographic forward simulation and calculate the measured intensities.

        :param patterns: A (N, H, W) tensor of diffraction patterns in the batch.
        :param positions: A (N, 2) tensor of probe positions in pixels.
        :return: measured intensities (squared magnitudes).
        """
        y = 0.0
        obj_patches = self.object.extract_patches(positions, self.probe.get_spatial_shape())
        for i_probe_mode in range(self.probe.n_modes):
            p = self.probe.get_mode(i_probe_mode)
            psi = obj_patches * p
            psi_far = torch.fft.fft2(psi, norm='ortho')
            psi_far = torch.fft.fftshift(psi_far, dim=(-2, -1))
            y = y + torch.abs(psi_far) ** 2
            
        # Revert the 1 / n factor due to batch-averaged loss
        # TODO: theoretically the gradient should be multiplied by |p|^2 / max(|p|^2) to be
        # equal to ePIE update function, but it doesn't seem to be the case. Need to investigate.
        psi.register_hook(lambda grad: grad * float(obj_patches.shape[0]))
        
        returns = [y]
        if return_object_patches:
            returns.append(obj_patches)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns
