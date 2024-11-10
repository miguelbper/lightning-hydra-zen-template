from enum import Enum, auto

from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor, sigmoid, softmax

Batch = tuple[Tensor, Tensor]
Metrics = dict[str, Tensor]


class Objects(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg: DictConfig
    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer


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
