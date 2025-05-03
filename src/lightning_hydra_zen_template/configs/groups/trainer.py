from hydra_zen import make_config, store
from lightning.pytorch import Trainer

from lightning_hydra_zen_template.configs.groups.callbacks import (
    EarlyStoppingCfg,
    ModelCheckpointCfg,
    RichModelSummaryCfg,
    RichProgressBarCfg,
)
from lightning_hydra_zen_template.configs.groups.logger import CSVLoggerCfg, MLFlowLoggerCfg, TensorBoardLoggerCfg
from lightning_hydra_zen_template.configs.groups.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation

TrainerDefaultCfg = fbuilds(
    Trainer,
    logger=[
        CSVLoggerCfg,
        TensorBoardLoggerCfg,
        MLFlowLoggerCfg,
    ],
    callbacks=[
        RichProgressBarCfg,
        RichModelSummaryCfg,
        EarlyStoppingCfg,
        ModelCheckpointCfg,
    ],
    min_epochs=1,
    max_epochs=10,
    check_val_every_n_epoch=1,
    deterministic=False,
    default_root_dir=output_dir,
    enable_model_summary=False,
    zen_wrappers=log_instantiation,
)

TrainerCPUCfg = make_config(
    accelerator="cpu",
    devices=1,
    bases=(TrainerDefaultCfg,),
)

TrainerGPUCfg = make_config(
    accelerator="gpu",
    devices=1,
    bases=(TrainerDefaultCfg,),
)

TrainerMPSCfg = make_config(
    accelerator="mps",
    devices=1,
    bases=(TrainerDefaultCfg,),
)

TrainerDDPSimCfg = make_config(
    accelerator="cpu",
    devices=2,
    strategy="ddp_spawn",
    bases=(TrainerDefaultCfg,),
)

TrainerDDPCfg = make_config(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    num_nodes=1,
    sync_batchnorm=True,
    bases=(TrainerDefaultCfg,),
)

trainer_store = store(group="trainer")
trainer_store(TrainerDefaultCfg, name="default")
trainer_store(TrainerCPUCfg, name="cpu")
trainer_store(TrainerGPUCfg, name="gpu")
trainer_store(TrainerMPSCfg, name="mps")
trainer_store(TrainerDDPSimCfg, name="ddp_sim")
trainer_store(TrainerDDPCfg, name="ddp")
