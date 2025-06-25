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
    """A PyTorch Lightning module that implements a complete training pipeline.

    This class handles the training, validation, and testing steps, as well as
    optimizer and learning rate scheduler configuration. It uses torchmetrics
    for tracking various metrics during training. Metrics are automatically
    cloned with appropriate prefixes ("val/" for validation, "test/" for testing)
    and logged during each step and epoch.

    Attributes:
        net (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        optimizer (Callable): Partially instantiated optimizer (remains to be instantiated with parameters).
        scheduler (Callable | None): Partially instantiated scheduler (remains to be instantiated with optimizer).
        metric_collection (MetricCollection): Collection of metrics to track.
        val_metrics (MetricCollection): Metrics for validation phase (cloned with "val/" prefix).
        test_metrics (MetricCollection): Metrics for testing phase (cloned with "test/" prefix).
    """

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

        Computes predictions, updates validation metrics, and logs them.
        Metrics are logged both per step and per epoch.

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

        Computes predictions, updates test metrics, and logs them.
        Metrics are logged both per step and per epoch.

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

        Creates the optimizer and optionally the learning rate scheduler.
        If a scheduler is provided, it's configured to update every epoch
        and monitor the training loss.

        Returns:
            OptimizerConfigType | OptimizerLRSchedulerConfigType: Configuration for the optimizer
                and optionally the learning rate scheduler. The scheduler (if present) is configured
                to update every epoch and monitor "train/loss".
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
