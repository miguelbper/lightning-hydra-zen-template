import os
from pathlib import Path

import pytest
import torch
from hydra_zen import launch, store, zen
from hydra_zen._launch import OverrideDict
from pytest import FixtureRequest

from lightning_hydra_zen_template.configs import TrainCfg
from lightning_hydra_zen_template.funcs.train import train


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


@pytest.fixture(params=[None, "high"])
def matmul_precision(request: FixtureRequest) -> str:
    return request.param


@pytest.fixture
def overrides(tmp_path: Path) -> OverrideDict:
    overrides = {
        "data.batch_size": 2,
        "data.num_workers": 0,
        "data.pin_memory": False,
        "hydra.run.dir": str(tmp_path),
        "trainer.accelerator": "cpu",
        "trainer.callbacks": None,
        "trainer.devices": 1,
        "trainer.limit_test_batches": 1,
        "trainer.limit_train_batches": 1,
        "trainer.limit_val_batches": 1,
        "trainer.logger": None,
        "trainer.max_epochs": 1,
    }
    return overrides


@pytest.fixture
def matrix_overrides(
    overrides: OverrideDict,
    accelerator: str,
    matmul_precision: str | None,
) -> OverrideDict:
    overrides = overrides.copy()
    overrides.update(
        {
            "matmul_precision": matmul_precision,
            "trainer.accelerator": accelerator,
        }
    )
    return overrides


@pytest.fixture
def ckpt_path(overrides: OverrideDict) -> str:
    overrides = overrides.copy()
    overrides.pop("trainer.callbacks")

    store.add_to_hydra_store()
    launch(TrainCfg, zen(train), version_base="1.3", overrides=overrides)
    ckpt_dir: str = os.path.join(overrides["hydra.run.dir"], "checkpoints", "last.ckpt")
    return ckpt_dir
