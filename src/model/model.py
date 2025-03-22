from collections.abc import Callable, Iterator
from typing import Any

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection

Input = Any
Target = Tensor
Batch = tuple[Input, Target]


class Model(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[[Iterator[Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler] | None,
        metric_collection: MetricCollection,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "loss_fn", "metric_collection"])

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_collection = metric_collection

        self.metrics = {
            "train": metric_collection.clone(prefix="train"),
            "val": metric_collection.clone(prefix="val"),
            "test": metric_collection.clone(prefix="test"),
        }

    def step(self, batch: Batch, batch_idx: int, split: str) -> Tensor:
        inputs, target = batch
        logits = self.model(inputs)
        loss = self.loss_fn(logits, target)
        self.log(f"{split}/loss", loss, on_step=True, on_epoch=False)
        self.metrics[split].update(logits, target)
        self.log_dict(self.metrics[split], on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self.step(batch, batch_idx, "val")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self.step(batch, batch_idx, "test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer: Optimizer = self.optimizer(params=self.parameters())
        optim_cfg = {"optimizer": optimizer}
        if self.scheduler:
            scheduler: LRScheduler = self.scheduler(optimizer=optimizer)
            optim_cfg["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            }
        return optim_cfg
