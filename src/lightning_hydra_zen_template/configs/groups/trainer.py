import os

from hydra_zen import store
from lightning.pytorch import Trainer

from lightning_hydra_zen_template.configs.utils.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.scikit_learn.checkpoint import SKLearnCheckpoint
from lightning_hydra_zen_template.scikit_learn.trainer import SKLearnTrainer

TrainerCfg = fbuilds(
    Trainer,
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)

SKLearnTrainerCfg = fbuilds(
    SKLearnTrainer,
    checkpoint_callback=fbuilds(
        SKLearnCheckpoint,
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor="${monitor}",
        mode="${mode}",
        zen_wrappers=log_instantiation,
    ),
    zen_wrappers=log_instantiation,
)

trainer_store = store(group="trainer")
trainer_store(TrainerCfg, name="default")
trainer_store(SKLearnTrainerCfg, name="sklearn")
