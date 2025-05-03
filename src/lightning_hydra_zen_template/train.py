import logging

import lightning as L
import torch
from hydra_zen import store, zen
from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.configs import TrainCfg
from lightning_hydra_zen_template.utils.print_config import print_config

log = logging.getLogger(__name__)


def train(
    data: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: str | None = None,
    evaluate: bool | None = True,
) -> float:
    log.info("Training model")
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)
    metric: torch.Tensor | None = trainer.checkpoint_callback.best_model_score
    ckpt_path: str = trainer.checkpoint_callback.best_model_path

    if evaluate and ckpt_path:
        log.info("Validating model")
        trainer.validate(model=model, datamodule=data, ckpt_path=ckpt_path)

        log.info("Testing model")
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)

    return metric.item() if metric is not None else None


def seed_fn(seed: int) -> None:
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def main() -> None:
    store(TrainCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=[zen(seed_fn), print_config])
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
