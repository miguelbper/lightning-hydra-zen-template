import importlib
import pkgutil

from lightning_hydra_zen_template.configs.eval import EvalCfg
from lightning_hydra_zen_template.configs.train import TrainCfg

__all__ = ["TrainCfg", "EvalCfg"]

packages = list(pkgutil.walk_packages(path=__path__, prefix=__name__ + "."))
modules = [module for module in packages if not module.ispkg]

# Import all modules in the configs directory, so that all configs are added to the store and available to hydra
for module in modules:
    importlib.import_module(name=f"{module.name}", package=__package__)
