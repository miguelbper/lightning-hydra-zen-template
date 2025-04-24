from hydra_zen import make_config

from src.configs.hydra_cfg import HydraCfg
from src.configs.trainer import TrainerCfg

RunCfg = make_config(
    task_name="mnist",
    evaluate=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
)


Config = make_config(
    datamodule=None,
    model=None,
    trainer=TrainerCfg,
    seed=42,
    bases=(HydraCfg, RunCfg),
)
