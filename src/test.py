import logging

import hydra
import rootutils
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

rootutils.setup_root(search_from=__file__, dotenv=False)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def test(cfg: DictConfig) -> None:
    """Test a model from a configuration object (which should include a
    checkpoint).

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        None
    """
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(cfg.trainer)

    log.info("Testing model")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    test()
