from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def instantiate_list(cfg: DictConfig | None) -> list[Any]:
    """Instantiate a list of objects based on the configuration dictionary.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        list[Any]: A list of instantiated objects.
    """
    cfg = cfg or OmegaConf.create({})
    return [instantiate(obj) for _, obj in cfg.items()]
