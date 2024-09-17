from typing import Type, Optional

import torch
import tqdm
from torch.utils.data import Dataset

from ptychotorch.data_structures import PtychographyVariableGroup
from ptychotorch.forward_models import ForwardModel
from ptychotorch.reconstructors.ad_general import AutodiffReconstructor


class AutodiffPtychographyReconstructor(AutodiffReconstructor):
    def __init__(self,
                 variable_group: PtychographyVariableGroup,
                 dataset: Dataset,
                 forward_model_class: Type[ForwardModel],
                 forward_model_params: Optional[dict] = None,
                 batch_size: int = 1,
                 loss_function: torch.nn.Module = None,
                 n_epochs: int = 100,
                 metric_function: Optional[torch.nn.Module] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(
            variable_group=variable_group,
            dataset=dataset,
            forward_model_class=forward_model_class,
            forward_model_params=forward_model_params,
            batch_size=batch_size,
            loss_function=loss_function,
            n_epochs=n_epochs,
            metric_function=metric_function,
            *args, **kwargs
        )
        
    def run_post_update_hooks(self) -> None:
        with torch.no_grad():
            if self.variable_group.object.optimizable:
                self.variable_group.object.post_update_hook()
            
            if self.variable_group.probe.optimizable and self.variable_group.opr_mode_weights.optimizable:
                weights = self.variable_group.probe.post_update_hook(self.variable_group.opr_mode_weights)
                self.variable_group.opr_mode_weights.set_data(weights)
            
            if self.variable_group.probe_positions.optimizable:
                self.variable_group.probe_positions.post_update_hook()
            
    