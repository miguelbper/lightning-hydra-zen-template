from hydra_zen import make_config

from src.configs.hydra_cfg import HydraCfg
from src.configs.trainer import TrainerCfg

RunCfg = make_config(  # TODO: Check how to declare that parameters here are missing, but required
    evaluate_after_train=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
)

Config = make_config(
    datamodule=None,  # TODO: Check how to declare that this parameter is missing, but required
    model=None,  # TODO: Check how to declare that this parameter is missing, but required
    trainer=TrainerCfg,
    seed=42,
    bases=(HydraCfg, RunCfg),
)
