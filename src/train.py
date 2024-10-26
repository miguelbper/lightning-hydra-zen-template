import logging
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.instantiate_list import instantiate_list
from src.utils.log_utils import log_cfg

Metrics = dict[str, float]
Objects = dict[str, Any]


rootutils.setup_root(__file__)
log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> tuple[Metrics, Objects]:
    """Train, validate, and optionally test a model using the provided configuration.

    The function performs the following steps:
    - Instantiates callbacks, loggers, model, datamodule, and trainer based on the configuration.
    - Trains the model using the provided datamodule and optionally resumes from a checkpoint.
    - Validates the model using the best checkpoint.
    - Optionally tests the model if evaluation is enabled in the configuration.
    - Returns the collected metrics and a dictionary of instantiated objects.

    Args:
        cfg (DictConfig): Configuration object containing all necessary parameters
                          for training, validation, and testing.
    Returns:
        tuple[Metrics, Objects]: A tuple containing the training metrics and a dictionary
                                 of instantiated objects used during the process.
    """
    # Set random seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate all objects
    log.info("Instantiating callbacks")
    callbacks: list[Callback] = instantiate_list(cfg.get("callbacks"))

    log.info("Instantiating loggers")
    logger: list[Logger] = instantiate_list(cfg.get("logger"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Log configuration
    log_cfg(cfg, trainer)

    # Train -> Validate -> Test
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    best_ckpt = trainer.checkpoint_callback.best_model_path

    trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    metrics = trainer.callback_metrics

    if cfg.get("eval"):
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
        metrics.update(trainer.callback_metrics)

    # Return metrics and objects
    objects = {
        "cfg": cfg,
        "callbacks": callbacks,
        "logger": logger,
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
    }
    return metrics, objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """
    Main function to train the model and return the specified metric.
    Args:
        cfg (DictConfig): Configuration object containing training parameters and settings.
    Returns:
        float | None: The value of the specified metric from the training process, or None if the
            metric is not found.
    """
    metrics, _ = train(cfg)
    return metrics.get(cfg.metric)


if __name__ == "__main__":
    main()
