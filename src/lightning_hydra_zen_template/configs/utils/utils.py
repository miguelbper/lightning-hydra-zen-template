import logging
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any

from hydra_zen import make_custom_builds_fn
from hydra_zen.wrapper import default_to_config
from omegaconf import DictConfig, OmegaConf
from toolz.functoolz import compose

log = logging.getLogger(__name__)


def log_instantiation(cfg: DictConfig) -> Callable[..., Any]:
    """Create a wrapper function that logs when a configuration is
    instantiated.

    This function creates a wrapper that logs the name of the configuration class
    being instantiated before actually instantiating it.

    Args:
        cfg (DictConfig): The configuration class to be instantiated.

    Returns:
        Callable[..., Any]: A wrapper function that logs and then instantiates the config.
    """

    def wrapper(*args, **kwargs):
        log.info(f"Instantiating <{cfg.__name__}>")
        return cfg(*args, **kwargs)

    return wrapper


def remove_types(cfg: DictConfig) -> DictConfig:
    """Remove type information from a configuration object.

    This function is used to allow multiple global configs to be used simultaneously
    by removing type information from the configuration object. In this repo, it is used
    to allow for experiment/debug/hparams_search configs to be used simultaneously.

    Implementation taken from https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621.

    Args:
        cfg (DictConfig): The configuration object to process.

    Returns:
        DictConfig: The configuration object with type information removed.
    """
    cfg = default_to_config(cfg)
    untype = compose(OmegaConf.create, OmegaConf.to_container, OmegaConf.create)
    return untype(cfg) if is_dataclass(cfg) else cfg


fbuilds = make_custom_builds_fn(populate_full_signature=True)
