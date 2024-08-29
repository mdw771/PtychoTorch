from typing import Type, Optional

import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import pandas as pd

from ptytorch.forward_models import ForwardModel, Ptychography2DForwardModel
from ptytorch.data_structures import VariableGroup, Ptychography2DVariableGroup


class LossTracker:
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.table = pd.DataFrame(columns=['epoch', 'loss'])

    def update(self, epoch: int, loss: float) -> None:
        self.table.loc[len(self.table)] = [epoch, loss]
        
    def print(self) -> None:
        print(self.table)
        
    def print_latest(self) -> None:
        print('Epoch: {}, Loss: {}'.format(
            int(self.table.iloc[-1].epoch), 
            self.table.iloc[-1].loss)
        )


class Reconstructor:
    
    def __init__(self, variable_group: VariableGroup):
        self.loss_tracker = LossTracker()
        self.variable_group = variable_group
    
    def build(self) -> None:
        pass
    
    
class IterativeReconstructor(Reconstructor):
    
    def __init__(self, 
                 variable_group: VariableGroup,
                 dataset: Dataset,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(variable_group, *args, **kwargs)
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.dataloader = None
        
    def build(self) -> None:
        super().build()
        self.build_dataloader()
        self.build_loss_tracker()
        
    def build_dataloader(self):
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size,
                                     generator=torch.Generator(device=torch.get_default_device()),
                                     shuffle=True)
    
    def build_loss_tracker(self):
        self.loss_tracker = LossTracker()


class AutodiffReconstructor(IterativeReconstructor):
    
    def __init__(self,
                 variable_group: VariableGroup,
                 dataset: Dataset,
                 forward_model_class: Type[ForwardModel],
                 forward_model_params: Optional[dict] = None,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            *args, **kwargs
        )
        self.forward_model_class = forward_model_class
        self.forward_model_params = forward_model_params if forward_model_params is not None else {}
        self.forward_model = None
        self.loss_function = loss_function
        
        self.check_inputs()
        
    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            assert var.optimizer is not None, \
                "Variable {} is optimizable but no optimizer is specified".format(var.name)
    
    def build(self) -> None:
        super().build()
        self.build_forward_model()
        
    def build_forward_model(self):
        self.forward_model = self.forward_model_class(self.variable_group, **self.forward_model_params)
        if not torch.get_default_device().type == 'cpu':
            self.forward_model = torch.nn.DataParallel(self.forward_model)
            self.forward_model.to(torch.get_default_device())
        
    def run(self, *args, **kwargs):
        for i_epoch in tqdm.trange(self.n_epochs):
            epoch_loss = 0.0
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())
                
                y_pred = self.forward_model(*input_data)
                batch_loss = self.loss_function(y_pred, y_true)
                batch_loss = self.loss_function(y_pred, y_true)

                batch_loss.backward()
                self.get_forward_model().post_differentiation_hook(*input_data, y_true)
                self.step_all_optimizers()
                self.forward_model.zero_grad()
                
                epoch_loss = epoch_loss + batch_loss.item()
            epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_tracker.update(epoch=i_epoch, loss=epoch_loss)
            self.loss_tracker.print_latest()
            
    def step_all_optimizers(self):
        for var in self.variable_group.get_optimizable_variables():
            var.optimizer.step()
            
    def get_forward_model(self) -> ForwardModel:
        if isinstance(self.forward_model, torch.nn.DataParallel):
            return self.forward_model.module
        else:
            return self.forward_model


class AnalyticalIterativeReconstructor(IterativeReconstructor):
    
    def __init__(self, 
        variable_group: VariableGroup,
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
        self.update_step_module: torch.nn.Module = None
        
    def build(self) -> None:
        super().build()
        self.build_update_step_module()
        
    def build_update_step_module(self, *args, **kwargs):
        update_step_func = self.compute_updates
        var_group = self.variable_group
        
        class EncapsulatedUpdateStep(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.variable_module_dict = torch.nn.ModuleDict(var_group.__dict__)
            
            def forward(self, *args, **kwargs):
                return update_step_func(self, *args, **kwargs)
            
        self.update_step_module = EncapsulatedUpdateStep()
        if not torch.get_default_device().type == 'cpu':
            self.update_step_module = torch.nn.DataParallel(self.update_step_module)
            self.update_step_module.to(torch.get_default_device())
        
    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module, *args, **kwargs) -> float:
        """
        Calculate the update vectors of optimizable variables that should be
        applied later. 
        
        This function will be encapsulated in a torch.nn.Module so that it works with
        DataParallel and DistributedDataParallel.
        
        Do not apply the update vectors to the optimizable variables in this function. 
        When called in a DataParallel or DistributedDataParallel context, the buffers
        of optimizable variables are copies so the updates will not persist. Instead,
        collect the returned update vectors, reduce them, and apply the updates in
        `apply_updates` which is called outside DataParallel. 
        
        When DataParallel is in use, the returned update vectors will have an additional
        dimension in the front corresponding to the number of replicas.
        
        :return: the update vectors.
        """
        raise NotImplementedError
    
    def apply_updates(self, *args, **kwargs):
        raise NotImplementedError


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
        self.check_inputs()
        
    def check_inputs(self, *args, **kwargs):
        for var in self.variable_group.get_optimizable_variables():
            assert 'lr' in var.optimizer_params.keys(), \
                "Optimizable variable {} must have 'lr' in optimizer_params.".format(var.name)
    
    def run(self, *args, **kwargs):
        torch.no_grad()
        for i_epoch in tqdm.trange(self.n_epochs):
            epoch_loss = 0.0
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())
                                
                (delta_o, delta_p), batch_loss = self.update_step_module(*input_data, y_true)
                self.apply_updates(input_data[0], delta_o, delta_p)
                batch_loss = torch.mean(batch_loss)
                                
                epoch_loss = epoch_loss + batch_loss.item()
            epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_tracker.update(epoch=i_epoch, loss=epoch_loss)
            self.loss_tracker.print_latest()
        
    @staticmethod
    def compute_updates(update_step_module: torch.nn.Module, 
                        indices: torch.Tensor, 
                        y_true: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
        object_ = update_step_module.variable_module_dict['object']
        probe = update_step_module.variable_module_dict['probe']
        probe_positions = update_step_module.variable_module_dict['probe_positions']
        
        assert probe.n_modes == 1, "This ePIE implementation only works with a single probe mode."
            
        indices = indices.cpu()
        positions = probe_positions.tensor[indices]
            
        y = 0.0
        obj_patches = object_.extract_patches(
            positions, probe.get_spatial_shape()
        )
        p = probe.get_mode(0)
        psi = obj_patches * p
        psi_far = torch.fft.fft2(psi, norm='ortho')
        psi_far = torch.fft.fftshift(psi_far, dim=(-2, -1))
        y = y + torch.abs(psi_far) ** 2
        
        psi_prime = psi_far / torch.abs(psi_far) * torch.sqrt(y_true + 1e-7)
        psi_prime = torch.fft.ifft2(torch.fft.ifftshift(psi_prime, dim=(-2, -1)), norm='ortho')
        
        delta_o = None
        if object_.optimizable:
            delta_o = p.conj() / (torch.abs(p) ** 2).max()
            delta_o = delta_o * (psi_prime - psi)
            # Add a new dimension in the front for DataParallel to concantenate them
            delta_o = delta_o[None, ...]
            # DataParallel would split the real and imaginary parts of delta_o
            # and store them in an additional dimension at the end. To keep things consistent,
            # we do the splitting manually for cases without DataParallel. 
            delta_o = torch.stack([delta_o.real, delta_o.imag], dim=-1)
        
        delta_p_all_modes = None
        if probe.optimizable:
            delta_p_all_modes = []
            delta_p = obj_patches.conj() / (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
            delta_p = delta_p * (psi_prime - psi)
            delta_p = delta_p.mean(0)
            delta_p_all_modes.append(delta_p)
            delta_p_all_modes = torch.cat(delta_p_all_modes, dim=0)
            # Add a new dimension in the front for DataParallel to concantenate them
            delta_p_all_modes = delta_p_all_modes[None, ...]
            delta_p_all_modes = torch.stack([delta_p_all_modes.real, delta_p_all_modes.imag], dim=-1)
            
        batch_loss = torch.mean((torch.sqrt(y) - torch.sqrt(y_true)) ** 2)
        return (delta_o, delta_p_all_modes), torch.atleast_1d(batch_loss)
    
    def apply_updates(self, indices, delta_o, delta_p, *args, **kwargs):
        object_ = self.variable_group.object
        probe = self.variable_group.probe
        probe_positions = self.variable_group.probe_positions

        if delta_o is not None:
            positions = probe_positions.tensor[indices]
            delta_o = delta_o[..., 0] + 1j * delta_o[..., 1]
            delta_o = delta_o.reshape(-1, delta_o.shape[-2], delta_o.shape[-1])
            delta_o = object_.optimizer_params['lr'] * delta_o
            self.variable_group.object.place_patches(positions, delta_o, op='add')
            
        if delta_p is not None:
            delta_p = delta_p[..., 0] + 1j * delta_p[..., 1]
            delta_p = torch.mean(delta_p, dim=0)
            delta_p = probe.optimizer_params['lr'] * delta_p
            p_new = self.variable_group.probe.get_mode(0) + delta_p
            self.variable_group.probe.tensor.set_data(p_new[None, :, :])
    