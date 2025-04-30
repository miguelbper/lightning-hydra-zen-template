import logging
import os

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from rootutils import setup_root

from lightning_hydra_zen_template.utils.print_config import save_and_print_cfg

root_dir = setup_root(search_from=__file__, dotenv=False, project_root_env_var=True)
config_dir = os.path.join(root_dir, "configs")
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=config_dir, config_name="config.yaml")
def train(cfg: DictConfig) -> float | None:
    """Train a model from a configuration object and return the specified
    metric.

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        float | None: The value of the specified metric from the training process
    """
    save_and_print_cfg(cfg)

    if cfg.get("seed"):
        log.info(f"Setting seed to {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True, verbose=False)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(cfg.trainer)

    log.info("Training model")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    metric: torch.Tensor | None = trainer.checkpoint_callback.best_model_score
    ckpt_path: str = trainer.checkpoint_callback.best_model_path

    if cfg.get("evaluate") and ckpt_path:
        log.info("Validating model")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        log.info("Testing model")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return metric.item() if metric is not None else None


if __name__ == "__main__":
    train()
