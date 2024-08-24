import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import pandas as pd

from ptychodus.api.reconstructor import ReconstructInput

from .forward_models import ForwardModel
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
        print(self.table.iloc[-1])


class Reconstructor:
    
    def __init__(self):
        self.loss_tracker = LossTracker()
    
    def build(self) -> None:
        pass


class AutodiffReconstructor(Reconstructor):
    
    def __init__(self,
                 dataset: Dataset,
                 forward_model: ForwardModel,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_model = forward_model
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.dataloader = None
    
    def build(self) -> None:
        super().build()
        self.build_dataloader()
        self.build_forward_model()
        
    def build_dataloader(self):
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size,
                                     generator=torch.Generator(device=torch.get_default_device()),
                                     shuffle=True)
        
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
        model = self.forward_model.module \
            if isinstance(self.forward_model, torch.nn.DataParallel) \
            else self.forward_model
        for var in model.optimizable_variables:
            var.optimizer.step()
