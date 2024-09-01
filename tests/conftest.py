import pytest
import rootutils
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train() -> DictConfig:
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train", return_hydra_config=True)
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".git"))
    return cfg


@pytest.fixture(scope="package")
def cfg_test() -> DictConfig:
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="test", return_hydra_config=True)
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".git"))
    return cfg
