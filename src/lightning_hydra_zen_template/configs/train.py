from hydra_zen import make_config

from lightning_hydra_zen_template.configs.utils.utils import add_colorlog, fbuilds
from lightning_hydra_zen_template.funcs.train import train

TrainCfg = make_config(
    bases=(fbuilds(train),),
    hydra_defaults=add_colorlog(
        [
            "_self_",
            {"data": "mnist"},
            {"model": "mnist"},
            {"trainer": "default"},
        ]
    ),
    seed=42,
    task_name="train",  # TODO: rename to experiment_name
    monitor="val/MulticlassAccuracy",
    mode="max",
)
