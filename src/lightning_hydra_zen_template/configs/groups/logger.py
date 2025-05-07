import os

from hydra_zen import make_config, store
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger

from lightning_hydra_zen_template.configs.groups.paths import log_dir, output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, remove_types

CSVLoggerCfg = fbuilds(
    CSVLogger,
    save_dir=output_dir,
    name="csv",
)

TensorBoardLoggerCfg = fbuilds(
    TensorBoardLogger,
    save_dir=output_dir,
    name="tensorboard",
)

MLFlowLoggerCfg = fbuilds(
    MLFlowLogger,
    tracking_uri=os.path.join(log_dir, "mlflow", "mlruns"),
)

LoggerDefaultCfg = make_config(
    logger=[
        CSVLoggerCfg,
        TensorBoardLoggerCfg,
        MLFlowLoggerCfg,
    ],
)

logger_store = store(group="logger", package="trainer", to_config=remove_types)
logger_store(LoggerDefaultCfg, name="default")
