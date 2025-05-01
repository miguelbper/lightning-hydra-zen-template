import logging
import os
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any

from hydra.conf import HydraConf, RunDir, SweepDir
from hydra_zen import make_config, make_custom_builds_fn, store
from hydra_zen.wrapper import default_to_config
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from omegaconf import OmegaConf
from rootutils import find_root
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)

from lightning_hydra_zen_template.data.mnist import MNISTDataModule
from lightning_hydra_zen_template.model.components.resnet import ResNet
from lightning_hydra_zen_template.model.model import Model

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------

log = logging.getLogger(__name__)


def log_instantiation(Cfg: Any) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        log.info(f"Instantiating <{Cfg.__name__}>")
        return Cfg(*args, **kwargs)

    return wrapper


# From https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621
# Purpose: have multiple _global_ configs which can be used simultaneously
# Example: at the same time, specify experiment and debug configs
def remove_types(x):
    x = default_to_config(x)
    if is_dataclass(x):
        # recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #           -> dict -> omegaconf dict (no types)
        return OmegaConf.create(OmegaConf.to_container(OmegaConf.create(x)))
    return x


builds = make_custom_builds_fn(populate_full_signature=True)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

root_dir = str(find_root(search_from=__file__))
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")
output_dir = "${hydra:runtime.output_dir}"
work_dir = "${hydra:runtime.cwd}"

year_month_day = "${now:%Y-%m-%d}"
hour_minute_second = "${now:%H-%M-%S}"
task_name = "${task_name}"

run_dir = os.path.join(log_dir, task_name, "runs", year_month_day, hour_minute_second)
sweep_dir = os.path.join(log_dir, task_name, "multiruns", year_month_day, hour_minute_second)
job_file = os.path.join(output_dir, f"{task_name}.log")

PathsCfg = make_config(
    root_dir=root_dir,
    data_dir=data_dir,
    log_dir=log_dir,
    output_dir=output_dir,
    work_dir=work_dir,
)

store(PathsCfg, name="default", group="paths")

# ------------------------------------------------------------------------------
# Hydra
# ------------------------------------------------------------------------------

HydraCfg = HydraConf(
    run=RunDir(run_dir),
    sweep=SweepDir(dir=sweep_dir, subdir="${hydra:job.num}"),
    # Fix from PR https://github.com/facebookresearch/hydra/pull/2242, while there isn't a new release
    job_logging={"handlers": {"file": {"filename": job_file}}},
)

store(HydraCfg)

# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------

RichProgressBarCfg = builds(RichProgressBar)
RichModelSummaryCfg = builds(RichModelSummary)
EarlyStoppingCfg = builds(
    EarlyStopping,
    monitor="${monitor}",
    patience=3,
    mode="${mode}",
)
ModelCheckpointCfg = builds(
    ModelCheckpoint,
    dirpath=os.path.join(output_dir, "checkpoints"),
    monitor="${monitor}",
    save_last=True,
    mode="${mode}",
    auto_insert_metric_name=False,
)
CallbacksDefaultCfg = make_config(
    rich_progress_bar=RichProgressBarCfg,
    rich_model_summary=RichModelSummaryCfg,
    early_stopping=EarlyStoppingCfg,
    model_checkpoint=ModelCheckpointCfg,
)

callbacks_store = store(group="callbacks")
callbacks_store(RichProgressBarCfg, name="rich_progress_bar")
callbacks_store(RichModelSummaryCfg, name="rich_model_summary")
callbacks_store(EarlyStoppingCfg, name="early_stopping")
callbacks_store(ModelCheckpointCfg, name="model_checkpoint")
callbacks_store(CallbacksDefaultCfg, name="default")

# ------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------

CSVLoggerCfg = builds(
    CSVLogger,
    save_dir=output_dir,
    name="csv",
)
TensorBoardLoggerCfg = builds(
    TensorBoardLogger,
    save_dir=output_dir,
    name="tensorboard",
)
MLFlowLoggerCfg = builds(
    MLFlowLogger,
    tracking_uri=os.path.join(log_dir, "mlflow", "mlruns"),
)

LoggerDefaultCfg = make_config(
    csv=CSVLoggerCfg,
    tensorboard=TensorBoardLoggerCfg,
    mlflow=MLFlowLoggerCfg,
)

logger_store = store(group="logger")
logger_store(CSVLoggerCfg, name="csv")
logger_store(TensorBoardLoggerCfg, name="tensorboard")
logger_store(MLFlowLoggerCfg, name="mlflow")
logger_store(LoggerDefaultCfg, name="default")

# ------------------------------------------------------------------------------
# DataModule
# ------------------------------------------------------------------------------

DataModuleCfg = builds(
    MNISTDataModule,
    data_dir=os.path.join(root_dir, "data"),
    zen_wrappers=log_instantiation,
)

store(DataModuleCfg, name="mnist", group="data")

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

NUM_CLASSES = 10

ModelCfg = builds(
    Model,
    net=builds(
        ResNet,
        num_classes=NUM_CLASSES,
    ),
    loss_fn=builds(
        nn.CrossEntropyLoss,
    ),
    optimizer=builds(
        Adam,
        zen_partial=True,
    ),
    scheduler=builds(
        ReduceLROnPlateau,
        mode="min",
        factor=0.1,
        patience=10,
        zen_partial=True,
    ),
    metric_collection=builds(
        MetricCollection,
        metrics=[
            builds(
                Accuracy,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="micro",
            ),
            builds(
                F1Score,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            builds(
                Precision,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            builds(
                Recall,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
        ],
    ),
    zen_wrappers=log_instantiation,
)

store(ModelCfg, name="mnist", group="model")

# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------

TrainerDefaultCfg = builds(
    Trainer,
    logger=[
        "${logger.csv}",
        "${logger.tensorboard}",
        "${logger.mlflow}",
    ],
    callbacks=[
        "${callbacks.rich_progress_bar}",
        "${callbacks.rich_model_summary}",
        "${callbacks.early_stopping}",
        "${callbacks.model_checkpoint}",
    ],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)

TrainerCPUCfg = make_config(
    accelerator="cpu",
    devices=1,
    bases=(TrainerDefaultCfg,),
)

TrainerGPUCfg = make_config(
    accelerator="gpu",
    devices=1,
    bases=(TrainerDefaultCfg,),
)

TrainerMPSCfg = make_config(
    accelerator="mps",
    devices=1,
    bases=(TrainerDefaultCfg,),
)

TrainerDDPSimCfg = make_config(
    accelerator="cpu",
    devices=2,
    strategy="ddp_spawn",
    bases=(TrainerDefaultCfg,),
)

TrainerDDPCfg = make_config(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    num_nodes=1,
    sync_batchnorm=True,
    bases=(TrainerDefaultCfg,),
)

store(TrainerDefaultCfg, name="default", group="trainer")
store(TrainerCPUCfg, name="cpu", group="trainer")
store(TrainerGPUCfg, name="gpu", group="trainer")
store(TrainerMPSCfg, name="mps", group="trainer")
store(TrainerDDPSimCfg, name="ddp_sim", group="trainer")
store(TrainerDDPCfg, name="ddp", group="trainer")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

TrainCfg = make_config(
    hydra_defaults=[
        "_self_",
        #
        {"data": "mnist"},
        {"model": "mnist"},
        {"callbacks": "default"},
        {"logger": "default"},
        {"trainer": "default"},
        {"paths": "default"},
        #
        {"experiment": None},
        {"hparams_search": None},
        {"debug": None},
        #
        {"override hydra/hydra_logging": "colorlog"},
        {"override hydra/job_logging": "colorlog"},
    ],
    #
    data=None,
    model=None,
    callbacks=None,
    logger=None,
    trainer=None,
    paths=None,
    #
    task_name="train",
    tags=["dev"],
    evaluate=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
)
