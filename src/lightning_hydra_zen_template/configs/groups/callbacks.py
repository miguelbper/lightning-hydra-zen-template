import os

from hydra_zen import make_config, store
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar

from lightning_hydra_zen_template.configs.utils.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, remove_types
from lightning_hydra_zen_template.utils.logging import LogConfigToMLflow

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
)

LogConfigToMLflowCfg = fbuilds(
    LogConfigToMLflow,
)


CallbacksTrainCfg = make_config(
    callbacks=[
        RichProgressBarCfg,
        RichModelSummaryCfg,
        EarlyStoppingCfg,
        ModelCheckpointCfg,
        LogConfigToMLflowCfg,
    ],
)

CallbacksEvalCfg = make_config(
    callbacks=[
        RichProgressBarCfg,
    ],
)


callbacks_store = store(group="callbacks", package="trainer", to_config=remove_types)
callbacks_store(CallbacksTrainCfg, name="train")
callbacks_store(CallbacksEvalCfg, name="eval")
