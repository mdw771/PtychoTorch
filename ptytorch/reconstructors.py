import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import pandas as pd

from ptychodus.api.reconstructor import ReconstructInput

from .forward_models import ForwardModel, Ptychography2DForwardModel
from .io_handles import PtychographyDataset
from .utils import set_default_complex_dtype


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
    
    def __init__(self):
        self.loss_tracker = LossTracker()
    
    def build(self) -> None:
        pass
    
    
class IterativeReconstructor(Reconstructor):
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
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
                 dataset: Dataset,
                 forward_model: ForwardModel,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            *args, **kwargs
        )
        self.forward_model = forward_model
        self.loss_function = loss_function
    
    def build(self) -> None:
        super().build()
        self.build_forward_model()
        
    def build_forward_model(self):
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

                batch_loss.backward()
                self.step_all_optimizers()
                self.forward_model.zero_grad()
                
                epoch_loss = epoch_loss + batch_loss.item()
            epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_tracker.update(epoch=i_epoch, loss=epoch_loss)
            self.loss_tracker.print_latest()
            
    def step_all_optimizers(self):
        for var in self.get_forward_model().optimizable_variables:
            var.optimizer.step()
            
    def get_forward_model(self) -> ForwardModel:
        if isinstance(self.forward_model, torch.nn.DataParallel):
            return self.forward_model.module
        else:
            return self.forward_model


class EPIEReconstructor(IterativeReconstructor):
    
    def __init__(self, 
                 dataset: Dataset,
                 forward_model: Ptychography2DForwardModel,
                 batch_size: int = 1,
                 n_epochs: int = 100,
                 alpha: float = 0.1,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            n_epochs=n_epochs,
            *args, **kwargs)
        self.alpha = alpha
        self.forward_model = forward_model
    
    def build(self) -> None:
        super().build()
        self.build_forward_model()
        
    def build_forward_model(self):
        self.forward_model = torch.nn.DataParallel(self.forward_model)
        self.forward_model.to(torch.get_default_device())
        
    def run(self, *args, **kwargs):
        torch.no_grad()
        for i_epoch in tqdm.trange(self.n_epochs):
            epoch_loss = 0.0
            for batch_data in self.dataloader:
                input_data = [x.to(torch.get_default_device()) for x in batch_data[:-1]]
                y_true = batch_data[-1].to(torch.get_default_device())
                                
                batch_loss = self.update_step(input_data[0], y_true)
                
                epoch_loss = epoch_loss + batch_loss.item()
            epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_tracker.update(epoch=i_epoch, loss=epoch_loss)
            self.loss_tracker.print_latest()
        
    def update_step(self, positions, y_true) -> float:
        # single mode only
        y = 0.0
        obj_patches = self.get_forward_model().object.extract_patches(
            positions, self.get_forward_model().probe.get_spatial_shape()
        )
        p = self.get_forward_model().probe.get_mode(0)
        psi = obj_patches * p
        psi_far = torch.fft.fft2(psi)
        psi_far = torch.fft.fftshift(psi_far, dim=(-2, -1))
        y = y + torch.abs(psi_far) ** 2
        
        psi_prime = psi_far / torch.abs(psi_far) * torch.sqrt(y_true + 1e-7)
        psi_prime = torch.fft.ifft2(torch.fft.ifftshift(psi_prime, dim=(-2, -1)))
        
        delta_o = self.alpha * p.conj() / (torch.abs(p) ** 2).max()
        delta_o = delta_o * (psi_prime - psi)
        
        self.get_forward_model().object.place_patches(positions, delta_o, op='add')
        
        delta_p = self.alpha * obj_patches.conj() / (torch.abs(obj_patches) ** 2).max(-1).values.max(-1).values.view(-1, 1, 1)
        delta_p = delta_p * (psi_prime - psi)
        delta_p = delta_p.mean(0)
        p = p + delta_p
        self.get_forward_model().probe.tensor.set_data(p[None, :, :])
        
        return torch.mean((torch.sqrt(y + 1e-7) - torch.sqrt(y_true + 1e-7)) ** 2)
    
    def get_forward_model(self) -> ForwardModel:
        if isinstance(self.forward_model, torch.nn.DataParallel):
            return self.forward_model.module
        else:
            return self.forward_model
        