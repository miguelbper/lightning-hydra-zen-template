from typing import Any

from lightning import LightningModule
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

Batch = tuple[Any, Any]


class LightningModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, target = batch
        # output = self.model(inputs)
        pass

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, target = batch
        # output = self.model(inputs)
        pass

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer
