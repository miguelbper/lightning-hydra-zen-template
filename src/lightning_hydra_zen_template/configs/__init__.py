from pathlib import Path

from rootutils import find_root

from lightning_hydra_zen_template.configs.eval import EvalCfg
from lightning_hydra_zen_template.configs.train import TrainCfg
from lightning_hydra_zen_template.utils.imports import import_modules

__all__ = ["TrainCfg", "EvalCfg"]

configs_dir: Path = find_root() / "src" / "lightning_hydra_zen_template" / "configs"
import_modules(pkg_dir=configs_dir)
