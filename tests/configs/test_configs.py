from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiate_list import instantiate_list


def test_cfg_train(cfg_train: DictConfig) -> None:
    HydraConfig().set_config(cfg_train)

    callbacks = instantiate_list(cfg_train.get("callbacks"))
    logger = instantiate_list(cfg_train.get("logger"))
    model = instantiate(cfg_train.model)
    datamodule = instantiate(cfg_train.datamodule)
    trainer = instantiate(cfg_train.trainer, callbacks=callbacks, logger=logger)

    assert all(isinstance(callback, Callback) for callback in callbacks)
    assert all(isinstance(log, Logger) for log in logger)
    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(trainer, Trainer)


def test_cfg_eval(cfg_eval: DictConfig) -> None:
    HydraConfig().set_config(cfg_eval)

    logger = instantiate_list(cfg_eval.get("logger"))
    model = instantiate(cfg_eval.model)
    datamodule = instantiate(cfg_eval.datamodule)
    trainer = instantiate(cfg_eval.trainer, logger=logger)

    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(trainer, Trainer)
