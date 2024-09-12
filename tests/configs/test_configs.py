import hydra
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiate_list import instantiate_list


def test_cfg_train(cfg_train: DictConfig) -> None:
    HydraConfig().set_config(cfg_train)

    callbacks = instantiate_list(cfg_train.get("callbacks"))
    logger = instantiate_list(cfg_train.get("logger"))
    model = hydra.utils.instantiate(cfg_train.model)
    datamodule = hydra.utils.instantiate(cfg_train.datamodule)
    trainer = hydra.utils.instantiate(cfg_train.trainer, callbacks=callbacks, logger=logger)

    assert all(isinstance(callback, Callback) for callback in callbacks)
    assert all(isinstance(log, Logger) for log in logger)
    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(trainer, Trainer)


def test_cfg_test(cfg_test: DictConfig) -> None:
    HydraConfig().set_config(cfg_test)

    logger = instantiate_list(cfg_test.get("logger"))
    model = hydra.utils.instantiate(cfg_test.model)
    datamodule = hydra.utils.instantiate(cfg_test.datamodule)
    trainer = hydra.utils.instantiate(cfg_test.trainer, logger=logger)

    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(trainer, Trainer)
