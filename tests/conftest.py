import pytest
from hydra import compose, initialize
from omegaconf import DictConfig


@pytest.fixture(scope="package")
def cfg_train() -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name="train")
    return cfg


@pytest.fixture(scope="package")
def cfg_test() -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name="test")
    return cfg
