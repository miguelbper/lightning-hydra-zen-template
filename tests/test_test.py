from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.test import test as evaluate
from src.train import train


def test_train_and_test(cfg: DictConfig):
    HydraConfig().set_config(cfg)

    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.evaluate = False
    train(cfg)

    with open_dict(cfg):
        cfg.ckpt_path = f"{cfg.paths.output_dir}/checkpoints/last.ckpt"
    evaluate(cfg)
