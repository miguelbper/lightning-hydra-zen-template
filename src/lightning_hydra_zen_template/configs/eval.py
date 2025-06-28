from hydra_zen import make_config
from omegaconf import MISSING

from lightning_hydra_zen_template.configs.utils.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import add_colorlog, fbuilds
from lightning_hydra_zen_template.funcs.eval import evaluate

EvalCfg = make_config(
    bases=(fbuilds(evaluate),),
    hydra_defaults=add_colorlog(
        [
            "_self_",
            {"data": "mnist"},
            {"model": "resnet"},
            {"trainer": "default"},
            {"callbacks": "eval"},
        ]
    ),
    task_name="eval",
    ckpt_path=MISSING,
    output_dir=output_dir,
)
