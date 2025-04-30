import copy
import os

import rich
import rich.syntax
import rich.tree
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, flag_override

from lightning_hydra_zen_template.utils.ranked_logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config(cfg: DictConfig) -> None:
    file_path = os.path.join(cfg.paths.output_dir, "resolved_config.yaml")
    log.info(f"Saving config to {file_path}")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open(file_path, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))

    remove_packages = ["hydra", "paths", "logger", "callbacks"]
    for package in remove_packages:
        if package in cfg:
            cfg = copy.copy(cfg)
            with flag_override(cfg, ["struct", "readonly"], [False, False]):
                cfg.pop(package)

    tree = rich.tree.Tree("CONFIG")

    print_order = ["data", "model", "trainer"]
    queue = []

    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            log.warning(f"Field '{field}' not found in config. Skipping '{field}' config printing...")

    for field in cfg:
        if field not in queue:
            queue.append(field)

    for key in queue:
        value = cfg[key]
        is_dict = isinstance(value, DictConfig)
        convert_fn = OmegaConf.to_yaml if is_dict else str
        branch_content = convert_fn(value)
        branch = tree.add(key)
        branch.add(rich.syntax.Syntax(branch_content, "yaml", theme="gruvbox-dark"))

    rich.print(tree)
