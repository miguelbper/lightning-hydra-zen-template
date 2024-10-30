import logging
from typing import Any

import hydra
import rootutils
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiate_list import instantiate_list
from src.utils.log_utils import log_cfg

Metrics = dict[str, float]
Objects = dict[str, Any]


rootutils.setup_root(__file__)
log = logging.getLogger(__name__)


def test(cfg: DictConfig) -> tuple[Metrics, Objects]:
    """Evaluate the model using the provided configuration.

    The function performs the following steps:
        1. Instantiates callback objects from the configuration.
        2. Instantiates logger objects from the configuration.
        3. Instantiates the model from the configuration.
        4. Instantiates the data module from the configuration.
        5. Instantiates the trainer with the callbacks and loggers.
        6. Logs the configuration.
        7. Tests the model using the trainer and data module.
        8. Returns the evaluation metrics and a dictionary of instantiated objects.

    Args:
        cfg (DictConfig): Configuration object containing all necessary parameters.

    Returns:
        tuple[Metrics, Objects]: A tuple containing:
            - Metrics: Evaluation metrics obtained from the model testing.
            - Objects: A dictionary of instantiated objects including:
    """
    # Instantiate all objects
    log.info("Instantiating callbacks")
    callbacks: list[Callback] = instantiate_list(cfg.get("callbacks"))

    log.info("Instantiating loggers")
    logger: list[Logger] = instantiate_list(cfg.get("logger"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)

    log.info("Instantiating trainer")
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Log configuration
    log_cfg(cfg, trainer)

    log.info("Testing model")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metrics = trainer.callback_metrics
    objects = {
        "cfg": cfg,
        "callbacks": callbacks,
        "logger": logger,
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
    }
    return metrics, objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to evaluate the model based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for evaluation.

    Returns:
        None
    """
    test(cfg)


if __name__ == "__main__":
    main()
