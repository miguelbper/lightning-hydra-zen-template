from hydra_zen import make_config

from lightning_hydra_zen_template.configs.utils.paths import output_dir
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
            {"callbacks": "train"},
            {"logger": "train"},
        ]
    ),
    seed=42,
    task_name="train",
    monitor="val/MulticlassAccuracy",
    mode="max",
    output_dir=output_dir,
)
