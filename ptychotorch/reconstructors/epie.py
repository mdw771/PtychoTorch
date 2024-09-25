from typing import Optional

import torch
import tqdm
from torch.utils.data import Dataset

import ptychotorch.propagation as prop
from ptychotorch.data_structures import Ptychography2DVariableGroup
from ptychotorch.reconstructors.base import AnalyticalIterativeReconstructor
from ptychotorch.image_proc import place_patches_fourier_shift


class EPIEReconstructor(AnalyticalIterativeReconstructor):

    def __init__(self,
                 variable_group: Ptychography2DVariableGroup,
                 dataset: Dataset,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            *args, **kwargs)

    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            assert 'lr' in var.optimizer_params.keys(), \
                "Optimizable variable {} must have 'lr' in optimizer_params.".format(var.name)
        if self.metric_function is not None:
            raise NotImplementedError('EPIEReconstructor does not support metric function yet.')
        if self.variable_group.probe.has_multiple_opr_modes:
            raise NotImplementedError('EPIEReconstructor does not support multiple OPR modes yet.')
        if self.variable_group.probe.n_modes > 1:
            raise NotImplementedError('EPIEReconstructor does not support mixed state probe yet.')

    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        (delta_o, delta_p), batch_loss = self.update_step_module(*input_data, y_true, self.dataset.valid_pixel_mask)
        self.apply_updates(delta_o, delta_p)
        batch_loss = torch.mean(batch_loss)
        self.loss_tracker.update_batch_loss_with_value(batch_loss.item())

    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module,
                        indices: torch.Tensor,
                        y_true: torch.Tensor,
                        valid_pixel_mask: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
        """
        Calculates the updates of the whole object, the probe, and other variables.
        This function is called in self.update_step_module.forward. 
        """
        object_ = update_step_module.variable_module_dict['object']
        probe = update_step_module.variable_module_dict['probe']
        probe_positions = update_step_module.variable_module_dict['probe_positions']

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]

        y = 0.0
        obj_patches = object_.extract_patches(
            positions, probe.get_spatial_shape()
        )
        p = probe.get_mode_and_opr_mode(0, 0)
        psi = obj_patches * p
        psi_far = prop.propagate_far_field(psi)
        y = y + torch.abs(psi_far) ** 2

        psi_prime = psi_far / torch.abs(psi_far) * torch.sqrt(y_true + 1e-7)
        # Do not swap magnitude for bad pixels.
        psi_prime = torch.where(valid_pixel_mask.repeat(psi_prime.shape[0], 1, 1), psi_prime, psi_far)
        psi_prime = prop.back_propagate_far_field(psi_prime)

        delta_o_patches = None
        if object_.optimizable:
            delta_o_patches = p.conj() / (torch.abs(p) ** 2).max()
            delta_o_patches = delta_o_patches * (psi_prime - psi)
            delta_o = place_patches_fourier_shift(torch.zeros_like(object_.data), positions + object_.center_pixel, delta_o_patches, op='add')

        delta_p_all_modes = None
        if probe.optimizable:
            delta_p = obj_patches.conj() / (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
            delta_p = delta_p * (psi_prime - psi)
            delta_p = delta_p.mean(0)
            delta_p_all_modes = delta_p[None, None, :, :]
            delta_p_all_modes = torch.nn.functional.pad(
                delta_p_all_modes, 
                (0, 0, 0, 0, 0, probe.n_modes - delta_p_all_modes.shape[1], 0, probe.n_opr_modes - delta_p_all_modes.shape[0])
            )
            
        # DataParallel would split the real and imaginary parts of delta_o
        # and store them in an additional dimension at the end. To keep things consistent,
        # we do the splitting manually for cases without DataParallel. 
        # Also, add a new dimension in the front for DataParallel to concantenate multi-device outputs.
        delta_o, delta_p_all_modes = update_step_module.process_updates(delta_o, delta_p_all_modes)

        batch_loss = torch.mean((torch.sqrt(y) - torch.sqrt(y_true)) ** 2)
        return (delta_o, delta_p_all_modes), torch.atleast_1d(batch_loss)

    def apply_updates(self, delta_o, delta_p, *args, **kwargs):
        """
        Apply updates to optimizable parameters given the updates calculated by self.compute_updates.

        :param delta_o: A (n_replica, h, w, 2) tensor of object update vector.
        :param delta_p: A (n_replicate, n_opr_modes, n_modes, h, w, 2) tensor of probe update vector.
        """
        object_ = self.variable_group.object
        probe = self.variable_group.probe

        if delta_o is not None:
            delta_o = delta_o[..., 0] + 1j * delta_o[..., 1]
            delta_o = delta_o.sum(0)
            object_.set_grad(-delta_o)
            object_.optimizer.step()
            
        if delta_p is not None:
            delta_p = delta_p[..., 0] + 1j * delta_p[..., 1]
            delta_p = delta_p.mean(0)
            probe.set_grad(-delta_p)
            probe.optimizer.step()
