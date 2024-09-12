import re
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


def format(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Formats the keys of a dictionary by removing any invalid characters.

    Valid characters: a-z, A-Z, 0-9, _, -, ., /, and space (those accepted as keys in MLflow).

    Args:
        dictionary (dict[str, Any]): The dictionary to be formatted.
    Returns:
        dict[str, Any]: The formatted dictionary with valid keys.
    """
    valid_chars = re.compile(r"[^a-zA-Z0-9_\-./ ]")
    remove_invalid = lambda k: valid_chars.sub("", k)  # noqa: E731
    formatted = {remove_invalid(k): v for k, v in dictionary.items()}
    return formatted


def log_cfg(cfg: DictConfig, trainer: Trainer) -> None:
    """Logs the configuration parameters and hyperparameters.

    Args:
        cfg (DictConfig): The configuration parameters.
        trainer (Trainer): The trainer object.
    Returns:
        None
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = format(flatten(cfg_dict))
    for logger in trainer.loggers:
        logger.log_hyperparams(flat_cfg)
