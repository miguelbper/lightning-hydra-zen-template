import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from pytest import FixtureRequest

from src.train import train


@pytest.fixture(params=["cpu", "mps", "cuda"])
def accelerator(request: FixtureRequest) -> str:
    device: str = request.param  # type: ignore
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return device


class TestTrain:
    def test_fast_dev_run(self, cfg: DictConfig, accelerator: str):
        HydraConfig().set_config(cfg)

        with open_dict(cfg):
            cfg.datamodule.batch_size = 2
            cfg.trainer.accelerator = accelerator
            cfg.trainer.fast_dev_run = True

        train(cfg)

    def test_limit_batches(self, cfg: DictConfig, accelerator: str):
        HydraConfig().set_config(cfg)

        with open_dict(cfg):
            cfg.datamodule.batch_size = 2
            cfg.trainer.accelerator = accelerator
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.limit_test_batches = 1

        train(cfg)
