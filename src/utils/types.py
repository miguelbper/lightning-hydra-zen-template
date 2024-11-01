from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
from torch import Tensor

Batch = tuple[Tensor, Tensor]
Metrics = dict[str, Tensor]


class Objects(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg: DictConfig
    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer
