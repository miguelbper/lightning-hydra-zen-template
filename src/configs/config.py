from pathlib import Path

import rootutils
from hydra_zen import builds, make_config
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger

from src.datamodule.mnist import MNISTDataModule

root_dir = rootutils.find_root(search_from=__file__)


Paths = make_config(
    root_dir=str(root_dir),
    data_dir=str(root_dir / "data"),
    raw_data_dir=str(root_dir / "data" / "raw"),
    processed_data_dir=str(root_dir / "data" / "processed"),
    log_dir=str(root_dir / "logs"),
    output_dir="${hydra:runtime.output_dir}",
    work_dir="${hydra:runtime.cwd}",
)


DataModuleCfg = builds(
    MNISTDataModule,
    data_dir="${paths.processed_data_dir}",
)


CSVLoggerCfg = builds(
    CSVLogger,
    save_dir="${paths.output_dir}",
    name="csv",
)


MLFlowLoggerCfg = builds(
    MLFlowLogger,
    tracking_uri=str(Path("${paths.log_dir}") / "mlflow" / "mlruns"),
)


EarlyStoppingCfg = builds(
    EarlyStopping,
    monitor="${monitor}",
    patience=3,
    mode="${mode}",
)


ModelCheckpointCfg = builds(
    ModelCheckpoint,
    dirpath=str(Path("${paths.output_dir}") / "checkpoints"),
    monitor="${monitor}",
    save_last=True,
    mode="${mode}",
    auto_insert_metric_name=False,
)


RichProgressBarCfg = builds(RichProgressBar)


RichModelSummaryCfg = builds(
    RichModelSummary,
    max_depth=-1,
)


TrainerCfg = builds(
    Trainer,
    logger=[
        CSVLoggerCfg,
        MLFlowLoggerCfg,
    ],
    callbacks=[
        EarlyStoppingCfg,
        ModelCheckpointCfg,
        RichProgressBarCfg,
        RichModelSummaryCfg,
    ],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir="${paths.output_dir}",
)


RunConfig = make_config(
    evaluate_after_train=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
)


Config = make_config(
    paths=Paths,
    datamodule=DataModuleCfg,
    trainer=TrainerCfg,
    bases=(RunConfig,),
)
