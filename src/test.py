import logging
from pathlib import Path

from hydra_zen import ZenStore, zen
from lightning import LightningDataModule, LightningModule, Trainer

from src.configs.config import Config

log = logging.getLogger(__name__)


def test(
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: Path,
) -> None:
    """Test a model from a configuration object (which should include a
    checkpoint).

    Args:
        cfg (DictConfig): Configuration object representing the config files.

    Returns:
        None
    """
    log.info("Testing model")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


def main() -> None:
    store = ZenStore(deferred_hydra_store=False)
    store(Config, name="config")
    task_function = zen(test)
    task_function.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
