import copy
import logging
import os

from omegaconf import DictConfig, OmegaConf, flag_override

log = logging.getLogger(__name__)


def print_cfg(cfg: DictConfig) -> None:
    file_path = os.path.join(cfg.paths.output_dir, "resolved_config.log")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    remove_packages = ["hydra", "paths", "logger", "callbacks"]
    for package in remove_packages:
        if package in cfg:
            cfg = copy.copy(cfg)
            with flag_override(cfg, ["struct", "readonly"], [False, False]):
                cfg.pop(package)

    yaml_str = OmegaConf.to_yaml(cfg)

    log.info(f"Composed config saved to {file_path}")
    print(yaml_str)
    with open(file_path, "w") as f:
        f.write(yaml_str)
