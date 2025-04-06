import logging
from pathlib import Path

import lightning as L
import torch
from hydra_zen import ZenStore, zen
from lightning import LightningDataModule, LightningModule, Trainer

from src.configs.config import Config

log = logging.getLogger(__name__)


def train(
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: Path | None,
    evaluate: bool,
) -> float | None:
    """Train a model from a configuration object and return the specified
    metric.

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        float | None: The value of the specified metric from the training process
    """
    log.info("Training model")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    metric: torch.Tensor | None = trainer.checkpoint_callback.best_model_score
    ckpt_path: str = trainer.checkpoint_callback.best_model_path

    if evaluate and ckpt_path:
        log.info("Validating model")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        log.info("Testing model")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return metric.item() if metric is not None else None


def seed_fn(seed: int) -> None:
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def main() -> None:
    store = ZenStore(deferred_hydra_store=False)
    store(Config, name="config")
    task_fn = zen(train, pre_call=zen(seed_fn))
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
