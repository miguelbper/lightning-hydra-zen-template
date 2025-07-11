from collections.abc import Callable

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfigType, OptimizerLRSchedulerConfigType
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT
from torchmetrics import MetricCollection

from lightning_hydra_zen_template.utils.types import Batch


class Model(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[[ParamsT], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler] | None,
        metric_collection: MetricCollection,
    ) -> None:
        """Initialize the Model.

        Args:
            net (nn.Module): The neural network model to train.
            loss_fn (nn.Module): The loss function to use for training.
            optimizer (Callable[[ParamsT], Optimizer]): Function that creates an optimizer.
            scheduler (Callable[[Optimizer], LRScheduler] | None): Function that creates a learning rate scheduler.
            metric_collection (MetricCollection): Collection of metrics to track during training.
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "loss_fn", "metric_collection"])
        self.net = net
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
        """Perform a single training step.

        Args:
            batch (Batch): A tuple of (input, target) tensors.
            batch_idx (int): The index of the current batch.

        Returns:
            Tensor: The computed loss value.
        """
        inputs, target = batch
        logits = self.net(inputs)
        loss = self.loss_fn(logits, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Perform a single validation step.

        Args:
            batch (Batch): A tuple of (input, target) tensors.
            batch_idx (int): The index of the current batch.
        """
        inputs, target = batch
        logits = self.net(inputs)
        self.val_metrics.update(logits, target)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Perform a single test step.

        Args:
            batch (Batch): A tuple of (input, target) tensors.
            batch_idx (int): The index of the current batch.
        """
        inputs, target = batch
        logits = self.net(inputs)
        self.test_metrics.update(logits, target)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> OptimizerConfigType | OptimizerLRSchedulerConfigType:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            OptimizerConfig | OptimizerLRSchedulerConfig: Configuration for the optimizer
                and optionally the learning rate scheduler.
        """
        optimizer: Optimizer = self.optimizer(self.parameters())
        if self.scheduler:
            scheduler: LRScheduler = self.scheduler(optimizer)
            optim_scheduler_cfg: OptimizerLRSchedulerConfigType = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "train/loss",
                },
            }
            return optim_scheduler_cfg
        else:
            optim_cfg: OptimizerConfigType = {"optimizer": optimizer}
            return optim_cfg
