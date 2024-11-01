import logging

import hydra
import rootutils
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from src.utils.types import Metrics, Objects
from src.utils.utils import log_cfg

rootutils.setup_root(__file__)
log = logging.getLogger(__name__)


def evaluate(cfg: DictConfig) -> tuple[Metrics, Objects]:
    """Evaluate the model using the provided configuration.

    The function performs the following steps:
    - Instantiates model, datamodule, and trainer based on the configuration.
    - Logs the configuration.
    - Tests the model using the trainer and data module.
    - Returns the evaluation metrics and a dictionary of instantiated objects.

    Args:
        cfg (DictConfig): Configuration object containing all necessary parameters.

    Returns:
        tuple[Metrics, Objects]: A tuple containing:
            - Metrics: Evaluation metrics obtained from the model testing.
            - Objects: A dictionary of instantiated objects.
    """
    log.info(f"Instantiating model <{cfg.model._target_}>...")
    model: LightningModule = instantiate(cfg.model)
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>...")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>...")
    trainer: Trainer = instantiate(cfg.trainer)
    objects = Objects(cfg=cfg, model=model, datamodule=datamodule, trainer=trainer)

    log_cfg(cfg, trainer)

    log.info("Testing model...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    metrics = trainer.callback_metrics

    return metrics, objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="cfg.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to evaluate the model based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for evaluation.

    Returns:
        None
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()
