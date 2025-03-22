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


def test(cfg: DictConfig) -> tuple[Metrics, Objects]:
    """Test a model from a configuration object (which should include a
    checkpoint).

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        tuple[Metrics, Objects]: A tuple containing:
            - Metrics: A dictionary of metrics from testing
            - Objects: The objects (cfg, model, datamodule, trainer) used in the testing process
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
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        None
    """
    test(cfg)


if __name__ == "__main__":
    main()
