from typing import Type, Optional, Union, Literal
import logging

import torch
import tqdm
from torch.utils.data import Dataset

from ptytorch.reconstructors.base import AnalyticalIterativeReconstructor
from ptytorch.data_structures import Ptychography2DVariableGroup
from ptytorch.forward_models import Ptychography2DForwardModel, PtychographyGaussianNoiseModel, PtychographyPoissonNoiseModel
from ptytorch.metrics import MSELossOfSqrt
import ptytorch.propagation as prop
from ptytorch.image_proc import place_patches_fourier_shift


class LSQMLReconstructor(AnalyticalIterativeReconstructor):
    """
    The least square maximum likelihood (LSQ-ML) algorithm described in
    
    Odstrčil, M., Menzel, A., & Guizar-Sicairos, M. (2018). Iterative 
    least-squares solver for generalized maximum-likelihood ptychography. 
    Optics Express, 26(3), 3108–3123. doi:10.1364/oe.26.003108
    
    This implementation uses automatic differentiation to get necessary gradients,
    but other steps, including the solving of the step size, are done analytically.
    """

    def __init__(self,
                 variable_group: Ptychography2DVariableGroup,
                 dataset: Dataset,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 noise_model: Literal['gaussian', 'poisson'] = 'gaussian',
                 noise_model_params: Optional[dict] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            *args, **kwargs)
        self.forward_model = Ptychography2DForwardModel(variable_group, retain_intermediates=True)
        noise_model_params = noise_model_params if noise_model_params is not None else {}
        self.noise_model = {
            'gaussian': PtychographyGaussianNoiseModel, 
            'poisson': PtychographyPoissonNoiseModel
            }[noise_model](**noise_model_params)
        self.alpha_psi_far = 0.5
        
    def get_psi_far_step_size(self, y_pred, y_true):
        if isinstance(self.noise_model, PtychographyGaussianNoiseModel):
            alpha = 0.5
            return alpha  # Eq. 16
        elif isinstance(self.noise_model, PtychographyPoissonNoiseModel):
            xi = 1 - y_true / y_pred  # Eq. 17
            batchsize = y_true.shape[0]
            alpha = torch.sum(y_pred - xi * (y_true / (1 - self.alpha_psi_far * xi))) / batchsize
            alpha = alpha / (torch.sum(xi ** 2 * y_pred) / batchsize)
            return alpha
        
    def run_reciprocal_space_step(self, y_pred, y_true):
        """
        Run step 1 of LSQ-ML, which updates psi. 
        
        :return: (batch_size, n_probe_modes, h, w) complex tensor
        """
        # gradient as in Eq. 12a/b
        psi_far_0 = self.forward_model.intermediate_variables['psi_far']
        dl_dpsi_far = self.noise_model.backward_to_psi_far(y_pred, y_true, psi_far_0)
        self.alpha_psi_far = self.get_psi_far_step_size(y_pred, y_true)
        psi_far = psi_far_0 - self.alpha_psi_far * dl_dpsi_far  # Eq. 14
        
        psi_opt = prop.back_propagate_far_field(psi_far)
        return psi_opt
    
    def run_real_space_step(self, psi_opt):
        """
        Run step 2 of LSQ-ML, which updates the object, probe, and other variables
        using psi updated in step 1. 
        
        This step is independent of the likelihood function chosen through `loss_function`.
        
        :param psi_opt: a (batch_size, n_probe_modes, h, w) complex tensor. Should be 
            psi updated in step 1.
        """
        positions = self.forward_model.intermediate_variables['positions']
        psi_0 = self.forward_model.intermediate_variables['psi']
        # chi.shape == (batch_size, n_probe_modes, h, w)
        chi = psi_opt - psi_0  # Eq, 19
        obj_patches = self.forward_model.intermediate_variables['obj_patches']
        
        self.update_object_and_probe(chi, obj_patches, positions)
            
    def update_object_and_probe(self, chi, obj_patches, positions, delta=1e-5, gamma=1e-5):
        delta_p_i = self._calculate_probe_gradient(chi, obj_patches)  # Eq. 24a
        delta_o_i = self._calculate_object_patch_gradient(chi)
        alpha_o_i, alpha_p_i = self.calculate_object_and_probe_update_step_sizes(
            chi, obj_patches, delta_o_i, delta_p_i, gamma=gamma
        )
        
        if self.variable_group.probe.optimizable:
            delta_p_hat = self._calculate_probe_update_direction(delta_p_i, delta=delta)  # Eq. 25a
            self._apply_probe_update(alpha_p_i, delta_p_hat, obj_patches)
        
        if self.variable_group.object.optimizable:
            delta_o_hat = self._calculate_object_update_direction(delta_o_i, positions, delta=delta)
            self._apply_object_update(alpha_o_i, delta_o_hat, positions)
        
    def calculate_object_and_probe_update_step_sizes(self, chi, obj_patches, delta_o_i, delta_p_i, gamma=1e-5):
        """
        Jointly calculate the update step sizes for object and probe according to Eq. 22 of Odstrcil (2018).
        This routine builds a (batch_size, 2, 2) batch matrix, batch-invert them to get the update step sizes.
        """
        probe = self.variable_group.probe.tensor.complex()
        
        # Shape of delta_p_o/o_p:     (batch_size, n_probe_modes, h, w)
        delta_p_o = delta_p_i * obj_patches[:, None, :, :]
        delta_o_patches_p = delta_o_i[:, None, :, :] * probe
        
        # Shape of aij:               (batch_size,)
        a11 = (delta_o_patches_p.abs() ** 2).sum(-1).sum(-1).sum(-1) + gamma
        a12 = (delta_o_patches_p * delta_p_o.conj()).sum(-1).sum(-1).sum(-1)
        a21 = a12.conj()
        a22 = (delta_p_o.abs() ** 2).sum(-1).sum(-1).sum(-1) + gamma
        b1 = torch.real(delta_o_patches_p.conj() * chi).sum(-1).sum(-1).sum(-1)
        b2 = torch.real(delta_p_o.conj() * chi).sum(-1).sum(-1).sum(-1)
        
        a_mat = torch.stack([a11, a12, a21, a22], dim=1).view(-1, 2, 2)
        b_vec = torch.stack([b1, b2], dim=1).view(-1, 2).type(a_mat.dtype)
        alpha_vec = torch.linalg.solve(a_mat, b_vec)
        alpha_vec = alpha_vec.abs()
        
        alpha_o_i = alpha_vec[:, 0]
        alpha_p_i = alpha_vec[:, 1]
        
        alpha_o_i, alpha_p_i = self.adjust_object_and_probe_update_step_sizes(alpha_o_i, alpha_p_i)
        
        return alpha_o_i, alpha_p_i
    
    def adjust_object_and_probe_update_step_sizes(self, alpha_o_i, alpha_p_i, threshold_o=1e-8, threshold_p=1e-8):
        alpha_p_max = alpha_p_i.max()
        alpha_o_max = alpha_o_i.max()
        
        logging.debug('alpha_p_i: min={}, max={}'.format(alpha_p_i.min(), alpha_p_max))
        logging.debug('alpha_o_i: min={}, max={}'.format(alpha_o_i.min(), alpha_o_max))
        
        if alpha_o_max < threshold_o or alpha_p_max < threshold_p:
            logging.warning('Update step size of either object or probe is extremely small '
                            '(alpha_p_max={}, alpha_o_max={}). This can be caused by bad '
                            'initial guesses. For example, you would have alpha_p ~ 1 and alpha_o ~ 0 '
                            'if the initial object is all ones.'.format(alpha_p_max, alpha_o_max))
        if alpha_o_max < threshold_o:
            # TODO: need to figure out a better way to set this
            logging.info('alpha_o is too small. Setting alpha_o to 1e3.')
            alpha_o_i[...] = 1e3
        if alpha_p_max < threshold_p:
            logging.info('alpha_p is too small. Setting alpha_p to 1.')
            alpha_p_i[...] = 1.0
        return  alpha_o_i, alpha_p_i
        
        
    def update_probe(self, chi, obj_patches, delta=1e-5, gamma=1e-5):
        delta_p_i = self._calculate_probe_gradient(chi, obj_patches)  # Eq. 24a
        alpha_p_i = self._calculate_probe_update_step_size(delta_p_i, chi, obj_patches, gamma=gamma)  # Eq. 23a
        delta_p_hat = self._calculate_probe_update_direction(delta_p_i, delta=delta)  # Eq. 25a
        logging.info('alpha_p_i: min={}, max={}'.format(alpha_p_i.min(), alpha_p_i.max()))
        self._apply_probe_update(alpha_p_i, delta_p_hat, obj_patches)
        
    def _calculate_probe_gradient(self, chi, obj_patches):
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape of obj_patches:  (batch_size, h, w)
        # Shape of delta_p:      (batch_size, n_probe_modes, h, w)
        delta_p = chi * obj_patches.conj()[:, None, :, :]  # Eq. 24a
        return delta_p
        
    def _calculate_probe_update_step_size(self, delta_p, chi, obj_patches, gamma=1e-5):
        # Shape of delta_p:      (batch_size, n_probe_modes, h, w)
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape of obj_patches:  (batch_size, h, w)
        # Shape of alpha:        (batch_size, n_probe_modes, h, w)
        delta_p_o = delta_p * obj_patches[:, None, :, :]
        alpha_p_i = torch.real(delta_p_o.conj() * chi)
        # Shape of alpha:        (batch_size,)
        alpha_p_i = alpha_p_i.sum(-1).sum(-1).sum(-1)
        alpha_denom = (delta_p_o.abs() ** 2).sum(-1).sum(-1).sum(-1)
        alpha_p_i = alpha_p_i / (alpha_denom + gamma)  # Eq. 23a
        return alpha_p_i
        
    def _calculate_probe_update_direction(self, delta_p, delta=1e-5):
        object_ = self.variable_group.object.tensor.complex()
        # Sum over batch dimension
        # Shape of delta_p_hat:  (n_probe_modes, h, w)
        delta_p_hat = torch.sum(delta_p, dim=0)  # Eq. 25a
        delta_p_hat = delta_p_hat / ((object_.abs() ** 2).sum() + delta)
        return delta_p_hat
    
    def _apply_probe_update(self, alpha_p_i, delta_p_hat, obj_patches, delta=1e-5):
        # Shape of alpha_p_i:        (batch_size,)
        # Shape of delta_p_hat:      (n_probe_modes, h, w)
        # Shape of obj_patches:      (batch_size, h, w)
        # Shape of update_vec:       (batch_size, n_probe_modes, h, w)
        update_vec = delta_p_hat * obj_patches[:, None, :, :].abs() ** 2
        update_vec = update_vec * alpha_p_i[:, None, None, None]
        # Shape of update_vec:       (n_probe_modes, h, w)
        update_vec = update_vec.sum(0)
        # Shape of update_vec_denom: (h, w)
        update_vec_denom = (obj_patches.abs() ** 2).sum(0)
        update_vec = update_vec / (update_vec_denom + delta)
        self.variable_group.probe.tensor.set_data(
            self.variable_group.probe.tensor.complex() + update_vec
        )
    
    def update_object(self, chi, positions, delta=1e-5, gamma=1e-5):
        delta_o_i = self._calculate_object_patch_gradient(chi)
        alpha_o_i = self._calculate_object_update_step_size(delta_o_i, chi, gamma=gamma)
        delta_o_hat = self._calculate_object_update_direction(delta_o_i, positions, delta=delta)
        self._apply_object_update(alpha_o_i, delta_o_hat, positions)
        
        logging.debug('alpha_o_i: min={}, max={}'.format(alpha_o_i.min(), alpha_o_i.max()))
    
    def _calculate_object_patch_gradient(self, chi):
        # Shape of probe:        (n_probe_modes, h, w)
        probe = self.variable_group.probe.tensor.complex()
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape delta_o_patches: (batch_size, h, w)
        # Multiply and sum over probe mode dimension
        delta_o_patches = torch.sum(chi * probe.conj(), dim=1)  # Eq. 24b
        return delta_o_patches
    
    def _calculate_object_update_step_size(self, delta_o_patches, chi, gamma=1e-5):
        probe = self.variable_group.probe.tensor.complex()
        # Shape of delta_o_patches: (batch_size, h, w)
        # Shape of probe:           (batch_size, n_probe_modes, h, w)
        delta_o_patches_p = delta_o_patches[:, None, :, :] * probe
        # Shape of chi:             (batch_size, n_probe_modes, h, w)
        alpha_o_i = torch.real(delta_o_patches_p.conj() * chi)
        # Shape of alpha_o_i:       (batch_size,)
        alpha_o_i = alpha_o_i.sum(-1).sum(-1).sum(-1)
        alpha_denom = (delta_o_patches_p.abs() ** 2).sum(-1).sum(-1).sum(-1)
        alpha_o_i = alpha_o_i / (alpha_denom + gamma)  # Eq. 23b
        return alpha_o_i
    
    def _calculate_object_update_direction(self, delta_o_patches, positions, delta=1e-5):
        # Shape of probe:        (n_probe_modes, h, w)
        probe = self.variable_group.probe.tensor.complex()
        # Shape of chi:          (batch_size, n_probe_modes, h, w)
        # Shape delta_o_patches: (batch_size, h, w)
        # Multiply and sum over probe mode dimension
        delta_o_hat = self.variable_group.object.place_patches_on_empty_buffer(positions, delta_o_patches)  # Eq. 25b
        # Shape of delta_o_hat:  (h_whole, w_whole)
        delta_o_hat = delta_o_hat / ((probe.abs() ** 2).sum() + delta)
        return delta_o_hat
    
    def _apply_object_update(self, alpha_o_i, delta_o_hat, positions, delta=1e-5):
        # Shape of alpha_o_i:    (batch_size,)
        probe = self.variable_group.probe.tensor.complex()
        # Sum over probe modes
        probe_int = (probe.abs() ** 2).sum(0, keepdim=True)
        # Shape of probe_int:    (batch_size, h, w)
        probe_int = probe_int.repeat(len(alpha_o_i), 1, 1)
        alpha_probe_int = alpha_o_i.view(-1, 1, 1) * probe_int
        alpha_probe_map = place_patches_fourier_shift(image=torch.zeros_like(delta_o_hat), 
                                                      positions=positions + self.variable_group.object.center_pixel,
                                                      patches=alpha_probe_int,
                                                      op='add')
        probe_map = place_patches_fourier_shift(image=torch.zeros_like(delta_o_hat), 
                                                positions=positions + self.variable_group.object.center_pixel,
                                                patches=probe_int,
                                                op='add')
        update_vec = (alpha_probe_map * delta_o_hat) / (probe_map + delta)
        self.variable_group.object.set_data(self.variable_group.object.data + update_vec)
        return update_vec
        
    def run(self, *args, **kwargs):
        for i_epoch in tqdm.trange(self.n_epochs):
            epoch_loss = 0.0
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())

                with torch.no_grad():
                    y_pred = self.forward_model(*input_data)
                    batch_loss = self.noise_model.nll(y_pred, y_true)
                    
                    psi_opt = self.run_reciprocal_space_step(y_pred, y_true)
                    self.run_real_space_step(psi_opt)
                    
                    epoch_loss = epoch_loss + batch_loss.item()
            epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_tracker.update(epoch=i_epoch, loss=epoch_loss)
            self.loss_tracker.print_latest()
