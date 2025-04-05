import rootutils
from hydra_zen import make_config
from lightning import Trainer

root_dir = rootutils.find_root(search_from=__file__)


Paths = make_config(
    root_dir=str(root_dir),
    data_dir=str(root_dir / "data" / "processed"),
    output_dir="${hydra:runtime.output_dir}",
)

TrainerCfg = make_config(trainer=Trainer)

Config = make_config(
    paths=Paths,
    seed=42,
    bases=(TrainerCfg,),
)

pass
