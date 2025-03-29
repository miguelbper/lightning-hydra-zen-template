import logging

import hydra
import lightning as L
import rootutils
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=False)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def train(cfg: DictConfig) -> float | None:
    """Train a model from a configuration object and return the specified
    metric.

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        float | None: The value of the specified metric from the training process
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.datamodule)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(cfg.trainer)

    if trainer.checkpoint_callback is None or not isinstance(trainer.checkpoint_callback, ModelCheckpoint):
        log.info("No checkpoint callback found. Providing a checkpoint callback is required for training.")
        return None

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
