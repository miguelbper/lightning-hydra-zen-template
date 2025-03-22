import re
from functools import partial
from typing import Any

from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


def flatten(dictionary: dict, parent_key: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        dictionary (dict[str, Any]): The nested dictionary to be flattened
        parent_key (str, optional): The parent key to be used for the flattened keys.
            Defaults to "".

    Returns:
        dict[str, Any]: The flattened dictionary
    """
    items: list[tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten(value, parent_key=new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def containers_to_dict(container: dict | list | Any) -> dict | Any:
    """Converts lists to dicts of the form index:value, in a nested combination
    of dicts and lists.

    Args:
        container (dict | list | Any): The nested combination of lists and dicts,
            or just a value

    Returns:
        dict | Any: The converted container with lists transformed to dicts
    """
    if isinstance(container, dict):
        return {k: containers_to_dict(v) for (k, v) in container.items()}
    elif isinstance(container, list):
        return containers_to_dict(dict(enumerate(container)))
    else:
        return container


def log_cfg(cfg: DictConfig, trainer: Trainer) -> None:
    """Logs the configuration parameters and hyperparameters.

    Args:
        cfg (DictConfig): The configuration object to log
        trainer (Trainer): The trainer object containing loggers

    Returns:
        None
    """
    cfg_cont = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = containers_to_dict(cfg_cont)
    cfg_flat = flatten(cfg_dict)

    invalid_chars = re.compile(r"[^a-zA-Z0-9_\-.]")
    replace_invalid = partial(invalid_chars.sub, "_")
    cfg_formatted = {replace_invalid(k): v for k, v in cfg_flat.items()}

    for logger in trainer.loggers:
        logger.log_hyperparams(cfg_formatted)
