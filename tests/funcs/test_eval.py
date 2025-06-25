from hydra_zen import launch, zen

from lightning_hydra_zen_template.configs import EvalCfg
from lightning_hydra_zen_template.funcs.eval import evaluate


def test_eval(ckpt_path: str) -> None:
    launch(EvalCfg, zen(evaluate), version_base="1.3", ckpt_path=ckpt_path)
