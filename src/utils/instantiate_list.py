from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def instantiate_list(cfg: Optional[DictConfig]) -> list[Any]:
    """Instantiate a list of objects based on the configuration dictionary.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        list[Any]: A list of instantiated objects.
    """
    cfg = cfg or OmegaConf.create({})
    return [hydra.utils.instantiate(obj) for _, obj in cfg.items()]
