import re
from collections.abc import MutableMapping
from functools import partial
from typing import Any

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


def flatten(dictionary: MutableMapping, parent_key: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        dictionary (dict[str, Any]): The nested dictionary to be flattened.
        parent_key (str, optional): The parent key to be used for the flattened keys.
    Returns:
        dict[str, Any]: The flattened dictionary.
    Example:
        >>> nested_dict = {"a": {"b": 1, "c": {"d": 2}}}
        >>> flatten(nested_dict)
        {'a.b': 1, 'a.c.d': 2}
    """
    items: list[tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, parent_key=new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def format(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Formats the keys of a dictionary by removing any invalid characters.

    Args:
        dictionary (dict[str, Any]): The dictionary to be formatted.
    Returns:
        dict[str, Any]: The formatted dictionary with valid keys.
    """
    invalid_chars = re.compile(r"[^a-zA-Z0-9_\-.]")
    replace_invalid = partial(invalid_chars.sub, "_")
    formatted = {replace_invalid(k): v for k, v in dictionary.items()}
    return formatted


def log_cfg(cfg: DictConfig, trainer: Trainer) -> None:
    """Logs the configuration parameters and hyperparameters.

    Args:
        cfg (DictConfig): The configuration parameters.
        trainer (Trainer): The trainer object.
    Returns:
        None
    """
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    flat_cfg: dict[str, Any] = format(flatten(cfg_dict))
    for logger in trainer.loggers:
        logger.log_hyperparams(flat_cfg)
