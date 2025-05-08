from hydra_zen import make_config, store
from lightning.pytorch import Trainer

from lightning_hydra_zen_template.configs.groups.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation

TrainerDefaultCfg = fbuilds(
    Trainer,
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)

TrainerCPUCfg = make_config(
    bases=(TrainerDefaultCfg,),
    accelerator="cpu",
    devices=1,
)

TrainerGPUCfg = make_config(
    bases=(TrainerDefaultCfg,),
    accelerator="gpu",
    devices=1,
)

TrainerMPSCfg = make_config(
    bases=(TrainerDefaultCfg,),
    accelerator="mps",
    devices=1,
)

TrainerDDPSimCfg = make_config(
    bases=(TrainerDefaultCfg,),
    accelerator="cpu",
    devices=2,
    strategy="ddp_spawn",
)

TrainerDDPCfg = make_config(
    bases=(TrainerDefaultCfg,),
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    num_nodes=1,
    sync_batchnorm=True,
)

trainer_store = store(group="trainer")
trainer_store(TrainerDefaultCfg, name="default")
trainer_store(TrainerCPUCfg, name="cpu")
trainer_store(TrainerGPUCfg, name="gpu")
trainer_store(TrainerMPSCfg, name="mps")
trainer_store(TrainerDDPSimCfg, name="ddp_sim")
trainer_store(TrainerDDPCfg, name="ddp")
