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
    def wrapper(*args, **kwargs):
        log.info(f"Instantiating <{cfg.__name__}>")
        return cfg(*args, **kwargs)

    return wrapper


# From https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621
# Purpose: have multiple _global_ configs which can be used simultaneously
# Example: at the same time, specify experiment and debug configs
def remove_types(cfg: DictConfig) -> DictConfig:
    cfg = default_to_config(cfg)
    untype = compose(OmegaConf.create, OmegaConf.to_container, OmegaConf.create)
    return untype(cfg) if is_dataclass(cfg) else cfg


fbuilds = make_custom_builds_fn(populate_full_signature=True)
