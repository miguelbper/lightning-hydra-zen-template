from collections.abc import Hashable, MutableMapping
from typing import Any

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


def flatten(
    dictionary: dict[Hashable, Any], parent_key: str = "", separator: str = "."
) -> dict[Hashable, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        dictionary (dict[Hashable, Any]): The nested dictionary to be flattened.
        parent_key (str, optional): The parent key to be used for the flattened keys.
        separator (str, optional): The separator to be used between parent and child keys.
    Returns:
        dict[Hashable, Any]: The flattened dictionary.
    Example:
        >>> nested_dict = {'a': {'b': 1, 'c': {'d': 2}}}
        >>> flatten(nested_dict)
        {'a.b': 1, 'a.c.d': 2}
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, parent_key=new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def log_cfg(cfg: DictConfig, trainer: Trainer) -> None:
    """Logs the configuration parameters and hyperparameters.

    Args:
        cfg (DictConfig): The configuration parameters.
        trainer (Trainer): The trainer object.
    Returns:
        None
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = flatten(cfg_dict)
    for logger in trainer.loggers:
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams(flat_cfg)
