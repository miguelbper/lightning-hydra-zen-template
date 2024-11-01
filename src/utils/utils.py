import re
from collections.abc import MutableMapping
from functools import partial
from typing import Any

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from src.utils.types import Metrics


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


def format(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Formats the keys of a dictionary by replacing any invalid characters.

    :param dictionary: The dictionary to be formatted
    :type dictionary: dict[str, Any]
    :return: The formatted dictionary with valid keys
    :rtype: dict[str, Any]
    """
    invalid_chars = re.compile(r"[^a-zA-Z0-9_\-.]")
    replace_invalid = partial(invalid_chars.sub, "_")
    formatted = {replace_invalid(k): v for k, v in dictionary.items()}
    return formatted


def log_cfg(cfg: DictConfig, trainer: Trainer) -> None:
    """Logs the configuration parameters and hyperparameters.

    :param cfg: _description_
    :type cfg: DictConfig
    :param trainer: _description_
    :type trainer: Trainer
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = format(flatten(cfg_dict))
    for logger in trainer.loggers:
        logger.log_hyperparams(flat_cfg)


def metric_value(metrics: Metrics, metric_name: str | None) -> float | None:
    """Retrieve the value of a specified metric from a Metrics object.

    :param metrics: An object containing various metrics
    :type metrics: Metrics
    :param metric_name: The name of the metric to retrieve
    :type metric_name: str | None
    :return: The value of the specified metric as a float. Returns None
        if the metric_name is None or not found.
    :rtype: float | None
    """
    if metric_name is None:
        return None
    if metric_name not in metrics:
        return None
    return metrics[metric_name].item()
