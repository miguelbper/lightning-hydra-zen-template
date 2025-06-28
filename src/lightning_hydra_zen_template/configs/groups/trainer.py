from hydra_zen import store
from lightning.pytorch import Trainer

from lightning_hydra_zen_template.configs.utils.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation

TrainerCfg = fbuilds(
    Trainer,
    min_epochs=1,
    max_epochs=3,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)


store(TrainerCfg, group="trainer", name="default")
