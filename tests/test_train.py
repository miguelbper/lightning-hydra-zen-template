from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train


class TestTrain:
    def test_fast_dev_run(self, cfg: DictConfig):
        HydraConfig().set_config(cfg)
        with open_dict(cfg):
            cfg.trainer.fast_dev_run = True
        train(cfg)

    def test_limit_batches(self, cfg: DictConfig):
        HydraConfig().set_config(cfg)
        with open_dict(cfg):
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.limit_test_batches = 1
        train(cfg)
