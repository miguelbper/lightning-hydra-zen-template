import os

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from pytest import FixtureRequest

from lightning_hydra_zen_template.test import test as evaluate
from lightning_hydra_zen_template.train import train


@pytest.fixture(params=["cpu", "mps", "cuda"])
def accelerator(request: FixtureRequest) -> str:
    device: str = request.param
    if device != "cpu" and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping GPU tests on GitHub Actions")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return device


def test_train_and_test(cfg: DictConfig, accelerator: str):
    HydraConfig().set_config(cfg)

    model_checkpoint = {
        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
        "dirpath": f"{cfg.paths.output_dir}/checkpoints",
        "save_last": True,
    }

    with open_dict(cfg):
        cfg.trainer.accelerator = accelerator
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.limit_test_batches = 1
        cfg.evaluate = False
        cfg.trainer.callbacks = model_checkpoint
    train(cfg)

    with open_dict(cfg):
        cfg.ckpt_path = f"{cfg.paths.output_dir}/checkpoints/last.ckpt"

    evaluate(cfg)
