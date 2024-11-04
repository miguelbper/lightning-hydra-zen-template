from typing import Any

from lightning import LightningModule
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from src.utils.types import Batch


class LightningModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        metric_collection: MetricCollection,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_collection = metric_collection
        self.metrics = {
            "train": self.metric_collection.clone(prefix="train/"),
            "val": self.metric_collection.clone(prefix="val/"),
            "test": self.metric_collection.clone(prefix="test/"),
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: implement forward pass
        pass

    def step(self, batch: Batch, batch_idx: int, split: str) -> Tensor:
        inputs, target = batch
        logits = self.model(inputs)
        loss = self.loss_fn(logits, target)
        self.log(f"{split}/loss", loss, on_step=True, on_epoch=False)
        self.metrics[split](logits, target)
        self.log_dict(self.metrics[split], on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self.step(batch, batch_idx, "val")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self.step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Optimizer:
        # TODO: include scheduler
        return self.optimizer
