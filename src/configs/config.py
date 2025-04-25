import copy
import logging
import os
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any

from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.experimental.callback import Callback
from hydra_zen import MISSING, builds, make_config, store
from hydra_zen.wrapper import default_to_config
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf, flag_override
from rootutils import find_root

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------


def log_instantiation(Cfg: Any) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        log.info(f"Instantiating <{Cfg.__name__}>")
        return Cfg(*args, **kwargs)

    return wrapper


class PrintConfigCallback(Callback):
    def on_run_start(self, config: DictConfig, config_name: str | None) -> None:
        if "hydra" in config:
            config = copy.copy(config)
            with flag_override(config, ["struct", "readonly"], [False, False]):
                config.pop("hydra")
        log.info("Printing composed config")
        print(OmegaConf.to_yaml(config))


# From https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621
# Purpose: have multiple _global_ configs which can be used simultaneously
# Example: at the same time, specify experiment and debug configs
def remove_types(x):
    x = default_to_config(x)
    if is_dataclass(x):
        # recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #           -> dict -> omegaconf dict (no types)
        return OmegaConf.create(OmegaConf.to_container(OmegaConf.create(x)))  # type: ignore
    return x


global_store = store(package="_global_", to_config=remove_types)


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

root_dir = str(find_root(search_from=__file__))
data_dir = os.path.join(root_dir, "data", "processed")
log_dir = os.path.join(root_dir, "logs")

run_dir = os.path.join(log_dir, "${task_name}", "runs", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
sweep_dir = os.path.join(log_dir, "${task_name}", "multiruns", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
output_dir = "${hydra:runtime.output_dir}"
job_file = os.path.join(output_dir, "${task_name}.log")

# ------------------------------------------------------------------------------
# Hydra
# ------------------------------------------------------------------------------

HydraCfg = HydraConf(
    run=RunDir(run_dir),
    sweep=SweepDir(dir=sweep_dir, subdir="${hydra:job.num}"),
    callbacks={"print_config": builds(PrintConfigCallback)},
    # Fix from PR https://github.com/facebookresearch/hydra/pull/2242, while there isn't a new release
    job_logging={"handlers": {"file": {"filename": job_file}}},
)

store(HydraCfg)


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------


RichProgressBarCfg = builds(RichProgressBar)
RichModelSummaryCfg = builds(RichModelSummary, max_depth=-1)
EarlyStoppingCfg = builds(EarlyStopping, monitor="${monitor}", patience=3, mode="${mode}")
ModelCheckpointCfg = builds(
    ModelCheckpoint,
    dirpath=os.path.join(output_dir, "checkpoints"),
    monitor="${monitor}",
    save_last=True,
    mode="${mode}",
    auto_insert_metric_name=False,
)


# ------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------

CSVLoggerCfg = builds(CSVLogger, save_dir=output_dir, name="csv")
TensorBoardLoggerCfg = builds(TensorBoardLogger, save_dir=output_dir, name="tensorboard")
MLFlowLoggerCfg = builds(MLFlowLogger, tracking_uri=os.path.join(log_dir, "mlflow", "mlruns"))


# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------


TrainerCfg = builds(
    Trainer,
    logger=[CSVLoggerCfg, TensorBoardLoggerCfg, MLFlowLoggerCfg],
    callbacks=[EarlyStoppingCfg, ModelCheckpointCfg, RichProgressBarCfg, RichModelSummaryCfg],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)


# ------------------------------------------------------------------------------
# Primary Config
# ------------------------------------------------------------------------------


Config = make_config(
    evaluate=True,
    ckpt_path=None,
    seed=42,
    task_name=MISSING,
    monitor=MISSING,
    mode=MISSING,
    datamodule=MISSING,
    model=MISSING,
    trainer=TrainerCfg,
)


experiment_store = global_store(group="experiment")


# ------------------------------------------------------------------------------
# Debug
# ------------------------------------------------------------------------------


DebugCfg = make_config(
    task_name="debug",
    bases=(Config,),
)


debug_store = global_store(group="debug")

debug_store(DebugCfg, name="debug")
