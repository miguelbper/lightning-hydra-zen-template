from hydra_zen import make_config

from src.configs.hydra_cfg import HydraCfg

Config = make_config(
    seed=42,
    bases=(HydraCfg,),
)
