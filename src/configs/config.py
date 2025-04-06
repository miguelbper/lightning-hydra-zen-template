import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rootutils
from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.experimental.callback import Callback
from hydra_zen import builds, make_config
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf, flag_override
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

from src.datamodule.mnist import MNISTDataModule
from src.model.model import Model
from src.model.resnet import ResNet

root_dir = rootutils.find_root(search_from=__file__)
log = logging.getLogger(__name__)


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


DataModuleCfg = builds(
    MNISTDataModule,
    data_dir="${paths.data_dir}",
    zen_wrappers=log_instantiation,
)


ModelCfg = builds(
    Model,
    model=builds(
        ResNet,
        num_classes=10,
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
                num_classes=10,
                average="micro",
            ),
            builds(
                F1Score,
                task="multiclass",
                num_classes=10,
                average="macro",
            ),
            builds(
                Precision,
                task="multiclass",
                num_classes=10,
                average="macro",
            ),
            builds(
                Recall,
                task="multiclass",
                num_classes=10,
                average="macro",
            ),
        ],
    ),
    zen_wrappers=log_instantiation,
)


TrainerCfg = builds(
    Trainer,
    logger=[
        builds(
            CSVLogger,
            save_dir="${paths.output_dir}",
            name="csv",
        ),
        builds(
            MLFlowLogger,
            tracking_uri=str(Path("${paths.log_dir}") / "mlflow" / "mlruns"),
        ),
    ],
    callbacks=[
        builds(
            EarlyStopping,
            monitor="${monitor}",
            patience=3,
            mode="${mode}",
        ),
        builds(
            ModelCheckpoint,
            dirpath=str(Path("${paths.output_dir}") / "checkpoints"),
            monitor="${monitor}",
            save_last=True,
            mode="${mode}",
            auto_insert_metric_name=False,
        ),
        builds(RichProgressBar),
        builds(
            RichModelSummary,
            max_depth=-1,
        ),
    ],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir="${paths.output_dir}",
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)


HydraCfg = HydraConf(
    run=RunDir(str(Path("${paths.log_dir}") / "hydra" / "runs" / "${now:%Y-%m-%d}" / "${now:%H-%M-%S}")),
    sweep=SweepDir(
        dir=str(Path("${paths.log_dir}") / "hydra" / "multiruns" / "${now:%Y-%m-%d}" / "${now:%H-%M-%S}"),
        subdir="${hydra.job.num}",
    ),
    callbacks={"print_config": builds(PrintConfigCallback)},
    # Fix from this https://github.com/facebookresearch/hydra/pull/2242 PR, while there isn't a new release
    job_logging={"handlers": {"file": {"filename": str(Path("${hydra.runtime.output_dir}") / ".log")}}},
)


Paths = make_config(
    root_dir=str(root_dir),
    data_dir=str(root_dir / "data" / "processed"),
    log_dir=str(root_dir / "logs"),
    output_dir="${hydra:runtime.output_dir}",
    work_dir="${hydra:runtime.cwd}",
)


RunCfg = make_config(
    evaluate_after_train=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
)


Config = make_config(
    hydra_defaults=[
        {"override hydra/job_logging": "colorlog"},
        {"override hydra/hydra_logging": "colorlog"},
        "_self_",
    ],
    datamodule=DataModuleCfg,
    model=ModelCfg,
    trainer=TrainerCfg,
    paths=Paths,
    hydra=HydraCfg,
    bases=(RunCfg,),
)
