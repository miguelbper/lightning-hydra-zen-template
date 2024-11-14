from collections.abc import Callable, Iterator
from enum import Enum, auto
from typing import Any

from lightning import LightningModule
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor, nn, sigmoid, softmax
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection

Batch = tuple[Tensor, Tensor]


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Task(Enum):
    REGRESSION = auto()
    BINARY = auto()
    MULTICLASS = auto()


class RegressionOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preds: Tensor = Field(alias="logits")


class BinaryOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: Tensor
    probs: Tensor = Field(init=False)
    preds: Tensor = Field(init=False)

    @field_validator("probs", mode="before")
    @classmethod
    def compute_probs(cls, v, values):
        return sigmoid(values["logits"])

    @field_validator("preds", mode="before")
    @classmethod
    def compute_preds(cls, v, values):
        return (values["probs"] > 0.5).float()


class MulticlassOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: Tensor
    probs: Tensor = Field(init=False)
    preds: Tensor = Field(init=False)

    @field_validator("probs", mode="before")
    @classmethod
    def compute_probs(cls, v, values):
        return softmax(values["logits"], dim=1)

    @field_validator("preds", mode="before")
    @classmethod
    def compute_preds(cls, v, values):
        return values["probs"].argmax(dim=1)


class LightningModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Callable[[Iterator[Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        metric_collection: MetricCollection,
        task: Task,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_collection = metric_collection
        self.task = task

        self.metrics = {
            Split.TRAIN: metric_collection.clone(prefix=f"{Split.TRAIN.value}/"),
            Split.VAL: metric_collection.clone(prefix=f"{Split.VAL.value}/"),
            Split.TEST: metric_collection.clone(prefix=f"{Split.TEST.value}/"),
        }
        self.output_cls_name = {
            Task.REGRESSION: RegressionOutput,
            Task.BINARY: BinaryOutput,
            Task.MULTICLASS: MulticlassOutput,
        }[task]

    def forward(self, inputs: Tensor) -> RegressionOutput | BinaryOutput | MulticlassOutput:
        logits = self.model(inputs)
        return self.output_cls_name(logits=logits)

    def step(self, batch: Batch, batch_idx: int, split: Split) -> Tensor:
        inputs, target = batch
        logits = self.model(inputs)
        loss = self.loss_fn(logits, target)
        self.log(f"{split.value}/loss", loss, on_step=True, on_epoch=False)
        self.metrics[split].update(logits, target)
        self.log_dict(self.metrics[split], on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self.step(batch, batch_idx, Split.TRAIN)

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self.step(batch, batch_idx, Split.VAL)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self.step(batch, batch_idx, Split.TEST)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.optimizer(params=self.parameters())
        optim_cfg = {"optimizer": optimizer}
        if self.scheduler:
            scheduler = self.scheduler(optimizer=optimizer)
            optim_cfg["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            }
        return optim_cfg
