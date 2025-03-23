from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def test_cfg(cfg: DictConfig) -> None:
    HydraConfig().set_config(cfg)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    callbacks = [instantiate(callback) for callback in cfg.trainer.callbacks]
    loggers = [instantiate(logger) for logger in cfg.trainer.logger]

    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(trainer, Trainer)
    assert all(isinstance(callback, Callback) for callback in callbacks)
    assert all(isinstance(logger, Logger) for logger in trainer.loggers)
    assert len(loggers) == len(trainer.loggers) == len(cfg.trainer.logger)
