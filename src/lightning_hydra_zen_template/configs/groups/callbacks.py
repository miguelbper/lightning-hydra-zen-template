import os

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar

from lightning_hydra_zen_template.configs.groups.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds

RichProgressBarCfg = fbuilds(
    RichProgressBar,
)

RichModelSummaryCfg = fbuilds(
    RichModelSummary,
)

EarlyStoppingCfg = fbuilds(
    EarlyStopping,
    monitor="${monitor}",
    patience=3,
    mode="${mode}",
)

ModelCheckpointCfg = fbuilds(
    ModelCheckpoint,
    dirpath=os.path.join(output_dir, "checkpoints"),
    monitor="${monitor}",
    save_last=True,
    mode="${mode}",
    auto_insert_metric_name=False,
)
