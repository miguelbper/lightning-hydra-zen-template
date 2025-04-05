import logging

import lightning as L
from hydra_zen import ZenStore, zen
from lightning import LightningDataModule, LightningModule, Trainer

from src.configs.config import Config

log = logging.getLogger(__name__)


def train_mock(datamodule: LightningDataModule, model: LightningModule, trainer: Trainer) -> None:
    log.info("Training model")


def seed_fn(seed: int) -> None:
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def main() -> None:
    store = ZenStore(deferred_hydra_store=False)
    store(Config, name="config")
    pre_seed = zen(seed_fn)
    task_fn = zen(train_mock, pre_call=pre_seed)
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
