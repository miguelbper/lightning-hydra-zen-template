import os

from hydra_zen import make_config, store
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar

from lightning_hydra_zen_template.configs.groups.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, remove_types

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

CallbacksDefaultCfg = make_config(
    callbacks=[
        RichProgressBarCfg,
        RichModelSummaryCfg,
        EarlyStoppingCfg,
        ModelCheckpointCfg,
    ],
)

CallbacksEvalCfg = make_config(callbacks=[RichProgressBarCfg])

callbacks_store = store(group="callbacks", package="trainer", to_config=remove_types)
callbacks_store(CallbacksDefaultCfg, name="default")
callbacks_store(CallbacksEvalCfg, name="eval")
