import logging
from collections.abc import Callable
from typing import Any

from hydra_zen import builds
from lightning.pytorch import Trainer

from src.configs.callbacks import (
    EarlyStoppingCfg,
    ModelCheckpointCfg,
    RichModelSummaryCfg,
    RichProgressBarCfg,
)
from src.configs.logger import CSVLoggerCfg, MLFlowLoggerCfg
from src.configs.paths import output_dir

log = logging.getLogger(__name__)


def log_instantiation(Cfg: Any) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        log.info(f"Instantiating <{Cfg.__name__}>")
        return Cfg(*args, **kwargs)

    return wrapper


TrainerCfg = builds(
    Trainer,
    logger=[CSVLoggerCfg, MLFlowLoggerCfg],
    callbacks=[EarlyStoppingCfg, ModelCheckpointCfg, RichProgressBarCfg, RichModelSummaryCfg],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)
