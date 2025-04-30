import importlib
import pkgutil

from lightning_hydra_zen_template.configs.train import Config

packages = list(pkgutil.walk_packages(path=__path__, prefix=__name__ + "."))
modules = [module for module in packages if not module.ispkg]


for module in modules:
    importlib.import_module(name=f"{module.name}", package=__package__)


__all__ = ["Config"]
