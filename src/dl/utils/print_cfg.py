import copy
import logging
import os

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf, flag_override

log = logging.getLogger(__name__)


def print_cfg(cfg: DictConfig) -> None:
    file_path = os.path.join(cfg.paths.output_dir, "resolved_config.log")
    log.info(f"Saving config to {file_path}")

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    remove_packages = ["hydra", "paths", "logger", "callbacks"]
    for package in remove_packages:
        if package in cfg:
            cfg = copy.copy(cfg)
            with flag_override(cfg, ["struct", "readonly"], [False, False]):
                cfg.pop(package)

    tree = rich.tree.Tree("CONFIG")

    for key, value in cfg.items():
        is_dict = isinstance(value, DictConfig)
        convert_fn = OmegaConf.to_yaml if is_dict else str
        branch_content = convert_fn(value)
        branch = tree.add(key)
        branch.add(rich.syntax.Syntax(branch_content, "yaml", theme="gruvbox-dark"))

    rich.print(tree)
    with open(file_path, "w") as file:
        rich.print(tree, file=file)
