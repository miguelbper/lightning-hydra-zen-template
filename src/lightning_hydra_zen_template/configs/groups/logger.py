import os

from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger

from lightning_hydra_zen_template.configs.groups.paths import log_dir, output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds

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
