import os

from hydra_zen import builds
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger

from src.configs.paths import log_dir, output_dir

CSVLoggerCfg = builds(CSVLogger, save_dir=output_dir, name="csv")
MLFlowLoggerCfg = builds(MLFlowLogger, tracking_uri=os.path.join(log_dir, "mlflow", "mlruns"))
