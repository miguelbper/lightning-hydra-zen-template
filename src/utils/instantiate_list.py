from typing import Any

import hydra
from omegaconf import DictConfig


def instantiate_list(cfg: DictConfig) -> list[Any]:
    """Instantiate a list of objects based on the configuration dictionary.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        list[Any]: A list of instantiated objects.
    """
    return [hydra.utils.instantiate(obj) for _, obj in cfg.items()] if cfg else []
