import logging
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=False)
log = logging.getLogger(__name__)  # TODO: should I be using loguru instead?


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def train(cfg: DictConfig) -> float:
    """Train a model from a configuration object and return the specified
    metric.

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        float: The value of the specified metric from the training process
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating objects...")
    model: LightningModule = instantiate(cfg.model)
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    trainer: Trainer = instantiate(cfg.trainer)

    log.info("Training model...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    best_ckpt: Path = trainer.checkpoint_callback.best_model_path

    log.info("Validating model...")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    metrics: dict[str, torch.Tensor] = trainer.callback_metrics

    if cfg.get("eval"):
        log.info("Testing model...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
        metrics.update(trainer.callback_metrics)
        # TODO: Will trainer.test override metrics? Check if this is correct

    return metrics[cfg.get("metric")].item()


if __name__ == "__main__":
    train()
