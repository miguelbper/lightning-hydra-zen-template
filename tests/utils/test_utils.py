from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.utils import flatten, log_cfg


class TestFlatten:
    def test_flatten_empty(self) -> None:
        empty: dict[str, int] = {}
        assert flatten(empty) == empty

    def test_flatten_simple(self) -> None:
        single_level = {"a": 1, "b": 2}
        assert flatten(single_level) == single_level

    def test_flatten_nested(self) -> None:
        nested = {"a": {"b": {"c": 1}}}
        assert flatten(nested) == {"a.b.c": 1}


class TestFormat:
    def test_format_empty(self) -> None:
        empty = {}
        assert format(empty) == empty

    def test_format_(self) -> None:
        empty = {}
        assert format(empty) == empty


class TestLogCfg:
    def test_log_cfg(self, cfg: DictConfig) -> None:
        HydraConfig().set_config(cfg)
        trainer = instantiate(cfg.trainer)
        log_cfg(cfg, trainer)
