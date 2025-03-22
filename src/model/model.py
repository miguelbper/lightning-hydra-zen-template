from collections.abc import Callable
from typing import Any

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT
from torchmetrics import MetricCollection

Input = Any
Target = Tensor
Batch = tuple[Input, Target]


class Model(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[[ParamsT], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler] | None,
        metric_collection: MetricCollection,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "loss_fn", "metric_collection"])
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Useful documentation for metrics
        # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#torchmetrics.MetricCollection
        # https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        self.metric_collection = metric_collection
        self.val_metrics = metric_collection.clone(prefix="val/")
        self.test_metrics = metric_collection.clone(prefix="test/")

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        inputs, target = batch
        logits = self.model(inputs)
        loss = self.loss_fn(logits, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)  # type: ignore
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, target = batch
        logits = self.model(inputs)
        self.val_metrics.update(logits, target)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)  # type: ignore

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, target = batch
        logits = self.model(inputs)
        self.test_metrics.update(logits, target)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)  # type: ignore

    def configure_optimizers(self) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        optimizer: Optimizer = self.optimizer(self.parameters())
        if self.scheduler:
            scheduler: LRScheduler = self.scheduler(optimizer)
            optim_scheduler_cfg: OptimizerLRSchedulerConfig = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val/loss",
                },
            }
            return optim_scheduler_cfg
        else:
            optim_cfg: OptimizerConfig = {"optimizer": optimizer}
            return optim_cfg
