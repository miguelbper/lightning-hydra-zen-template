import logging

import hydra
import lightning as L
import rootutils
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from src.utils.types import Metrics, Objects
from src.utils.utils import log_cfg, metric_value

rootutils.setup_root(__file__)
log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> tuple[Metrics, Objects]:
    """Train a model from a configuration object.

    :param cfg: Configuration object representing the config files.
    :type cfg: DictConfig
    :return: A dictionary of metrics and the objects (cfg, model,
        datamodule, trainer) used in the training process.
    :rtype: tuple[Metrics, Objects]
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

    log.info("Training model...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    best_ckpt = trainer.checkpoint_callback.best_model_path

    log.info("Validating model...")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    metrics = trainer.callback_metrics

    if cfg.get("eval"):
        log.info("Testing model...")
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
    metric = metric_value(metrics, cfg.get("metric"))
    return metric


if __name__ == "__main__":
    main()
