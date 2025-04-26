from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict

rootutils.setup_root(search_from=__file__, dotenv=False)


@pytest.fixture(scope="function")
def cfg(tmp_path: Path) -> DictConfig:
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=mnist"], return_hydra_config=True)

        with open_dict(cfg):
            cfg.hydra.runtime.output_dir = str(tmp_path)
            cfg.hydra.job.num = 1
            cfg.trainer.callbacks = None
            cfg.trainer.logger = None
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.datamodule.batch_size = 2

    return cfg
