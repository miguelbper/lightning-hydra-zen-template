import glob
import os

import hydra
import pytest
from omegaconf import OmegaConf

path_file = os.path.abspath(__file__)
path_root = os.path.dirname(os.path.dirname(os.path.dirname(path_file)))
path_conf = os.path.join(path_root, "configs")
yaml_files = glob.glob(os.path.join(path_conf, "**/*.yaml"), recursive=True)


@pytest.fixture(params=yaml_files, ids=map(os.path.basename, yaml_files))
def yaml_file(request) -> str:
    return request.param


def test_configs(yaml_file: str):
    cfg = OmegaConf.load(yaml_file)
    is_class = "_target_" in cfg
    no_missing = not any(OmegaConf.is_missing(cfg, key) for key in cfg.keys())
    if is_class and no_missing:
        obj = hydra.utils.instantiate(cfg)
        cls_name_requested = cfg._target_.split(".")[-1]
        cls_name_instantiated = obj.__class__.__name__
        assert cls_name_requested == cls_name_instantiated
