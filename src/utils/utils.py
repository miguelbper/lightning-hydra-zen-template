import re
from collections.abc import MutableMapping
from functools import partial
from typing import Any

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


def flatten(dictionary: MutableMapping, parent_key: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    :param dictionary: The nested dictionary to be flattened
    :type dictionary: MutableMapping
    :param parent_key: The parent key to be used for the flattened keys,
        defaults to ""
    :type parent_key: str, optional
    :return: The flattened dictionary
    :rtype: dict[str, Any]
    """
    items: list[tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, parent_key=new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def log_cfg(cfg: DictConfig, trainer: Trainer) -> None:
    """Logs the configuration parameters and hyperparameters.

    :param cfg: _description_
    :type cfg: DictConfig
    :param trainer: _description_
    :type trainer: Trainer
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_flat = flatten(cfg_dict)

    invalid_chars = re.compile(r"[^a-zA-Z0-9_\-.]")
    replace_invalid = partial(invalid_chars.sub, "_")
    cfg_formatted = {replace_invalid(k): v for k, v in cfg_flat.items()}

    for logger in trainer.loggers:
        logger.log_hyperparams(cfg_formatted)
