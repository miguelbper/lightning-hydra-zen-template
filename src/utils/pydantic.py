from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from pydantic import BaseModel


class Objects(BaseModel):
    cfg: DictConfig
    model: LightningModule
    datamodule: LightningDataModule
    trainer: Trainer
