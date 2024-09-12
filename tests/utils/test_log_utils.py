import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.utils.instantiate_list import instantiate_list
from src.utils.log_utils import flatten, log_cfg


class TestFlatten:
    def test_flatten_empty(self) -> None:
        empty = {}
        assert flatten(empty) == empty

    def test_flatten_simple(self) -> None:
        single_level = {"a": 1, "b": 2}
        assert flatten(single_level) == single_level

    def test_flatten_nested(self) -> None:
        nested = {"a": {"b": {"c": 1}}}
        assert flatten(nested) == {"a.b.c": 1}


class TestLogCfg:
    def test_log_cfg(self, cfg_train: DictConfig) -> None:
        HydraConfig().set_config(cfg_train)

        logger = instantiate_list(cfg_train.get("logger"))
        trainer = hydra.utils.instantiate(cfg_train.trainer, logger=logger)

        log_cfg(cfg_train, trainer)

        # for log in logger:
        #     if hasattr(log, "log_hyperparams"):
        #         assert log.log_hyperparams.called
        #         assert log.log_hyperparams.call_count == 1
        #         assert log.log_hyperparams.call_args[0][0] == flatten(cfg_train)
        #         assert log.log_hyperparams.call_args[1] == {}
