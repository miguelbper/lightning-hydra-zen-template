import logging

import hydra
import lightning as L
import rootutils
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from src.utils.log_utils import log_cfg
from src.utils.pydantic import Objects

Metrics = dict[str, float]  # TODO: float or Tensor?


rootutils.setup_root(__file__)
log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> tuple[Metrics, Objects]:
    """Train, validate, and optionally test a model using the provided configuration.

    The function performs the following steps:
    - Instantiates model, datamodule, and trainer based on the configuration.
    - Trains the model using the provided datamodule and optionally resumes from a checkpoint.
    - Validates the model using the best checkpoint.
    - Optionally tests the model if evaluation is enabled in the configuration.
    - Returns the collected metrics and a dictionary of instantiated objects.

    Args:
        cfg (DictConfig): Configuration object containing all necessary parameters
                          for training, validation, and testing.

    Returns:
        tuple[Metrics, Objects]: A tuple containing:
            Metrics: Collected metrics from the training process.
            Objects: Instantiated objects used during the process.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>...")
    model: LightningModule = instantiate(cfg.model)
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>...")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>...")
    trainer: Trainer = instantiate(cfg.trainer)
    objects = Objects(cfg=cfg, model=model, datamodule=datamodule, trainer=trainer)

    log_cfg(cfg, trainer)

    log.info("Training model")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    best_ckpt = trainer.checkpoint_callback.best_model_path

    log.info("Validating model")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    metrics = trainer.callback_metrics

    if cfg.get("eval"):
        log.info("Testing model")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
        metrics.update(trainer.callback_metrics)
        # TODO: Will trainer.test override metrics? Check if this is correct

    return metrics, objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="cfg.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main function to train the model and return the specified metric.

    Args:
        cfg (DictConfig): Configuration object containing training parameters and settings.

    Returns:
        float | None: The value of the specified metric from the training process, or None if the
            metric is not found.
    """
    metrics, _ = train(cfg)
    metric_name = cfg.get("metric") or ""
    metric_value = metrics.get(metric_name)
    return metric_value


if __name__ == "__main__":
    main()
