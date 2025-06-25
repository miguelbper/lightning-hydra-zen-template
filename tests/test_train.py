import subprocess
import sys
from typing import Any

from hydra_zen._launch import OverrideDict
from rootutils import find_root


def test_main(overrides: OverrideDict) -> None:
    def value_to_str(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    args = [f"{key}={value_to_str(value)}" for key, value in overrides.items()]
    train_script = find_root() / "src" / "lightning_hydra_zen_template" / "train.py"
    subprocess.run([sys.executable, str(train_script)] + args, check=True)
