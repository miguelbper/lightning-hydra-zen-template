import pytest
import rootutils
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__)
output = "${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}"


@pytest.fixture(scope="package")
def cfg() -> DictConfig:
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="cfg", return_hydra_config=True)
        with open_dict(cfg):
            cfg.hydra.job.num = 1
            cfg.hydra.runtime.output_dir = output
    return cfg
