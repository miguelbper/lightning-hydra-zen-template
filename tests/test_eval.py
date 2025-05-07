import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra_zen import launch, store, zen
from hydra_zen._launch import OverrideDict
from rootutils import find_root

from lightning_hydra_zen_template.configs import EvalCfg, TrainCfg
from lightning_hydra_zen_template.eval import evaluate
from lightning_hydra_zen_template.train import train


@pytest.fixture()
def overrides(tmp_path: Path) -> OverrideDict:
    overrides = {
        "data.batch_size": 2,
        "data.num_workers": 0,
        "data.pin_memory": False,
        "hydra.run.dir": str(tmp_path),
        "trainer.accelerator": "cpu",
        "trainer.devices": 1,
        "trainer.limit_test_batches": 1,
        "trainer.limit_train_batches": 1,
        "trainer.limit_val_batches": 1,
        "trainer.logger": None,
        "trainer.max_epochs": 1,
    }
    return overrides


@pytest.fixture()
def ckpt_path(overrides: OverrideDict) -> str:
    store.add_to_hydra_store()
    launch(TrainCfg, zen(train), version_base="1.3", overrides=overrides)
    ckpt_path = os.path.join(overrides["hydra.run.dir"], "checkpoints", "last.ckpt")
    return ckpt_path


class TestEval:
    def test_eval(self, ckpt_path: str) -> None:
        launch(EvalCfg, zen(evaluate), version_base="1.3", ckpt_path=ckpt_path)

    def test_main(self, ckpt_path: str) -> None:
        eval_script = find_root() / "src" / "lightning_hydra_zen_template" / "eval.py"
        subprocess.run([sys.executable, str(eval_script), f"ckpt_path={ckpt_path}"], check=True)
