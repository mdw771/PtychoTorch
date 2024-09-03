import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ptytorch.data_structures import VariableGroup


class LossTracker:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.table = pd.DataFrame(columns=['epoch', 'loss'])
        self.table['epoch'] = self.table['epoch'].astype(int)

    def update(self, epoch: int, loss: float) -> None:
        self.table.loc[len(self.table)] = [epoch, loss]

    def print(self) -> None:
        print(self.table)

    def print_latest(self) -> None:
        print('Epoch: {}, Loss: {}'.format(
            int(self.table.iloc[-1].epoch),
            self.table.iloc[-1].loss)
        )

    def to_csv(self, path: str) -> None:
        self.table.to_csv(path, index=False)


class Reconstructor:

    def __init__(self, variable_group: VariableGroup):
        self.loss_tracker = LossTracker()
        self.variable_group = variable_group

    def build(self) -> None:
        pass

    def get_config_dict(self) -> dict:
        d = self.variable_group.get_config_dict()
        return d


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

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d.update({'batch_size': self.batch_size, 'n_epochs': self.n_epochs})
        return d


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
            # TODO: use CUDA stream instead of DataParallel for non-AD reconstructor. 
            # https://poe.com/s/NZUVScEEGLxBE5ZDmKc0
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