import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig

from src.model.model import Model

B, C, H, W = 32, 3, 32, 32
num_classes = 10


@pytest.fixture
def model(cfg: DictConfig) -> Model:
    HydraConfig().set_config(cfg)
    model = instantiate(cfg.model)
    return model


class TestModel:
    def test_init(self, model: Model):
        assert isinstance(model, LightningModule)

    def test_configure_optimizers(self, model: Model):
        optim_cfg = model.configure_optimizers()
        assert isinstance(optim_cfg, dict)
        assert "optimizer" in optim_cfg

    def test_configure_optimizers_no_scheduler(self, model: Model):
        model.scheduler = None
        optim_cfg = model.configure_optimizers()
        assert isinstance(optim_cfg, dict)
        assert "optimizer" in optim_cfg

    def test_training_step(self, model: Model):
        batch = (torch.randn(B, C, H, W), torch.randint(0, num_classes, (B,)))
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    def test_validation_step(self, model: Model):
        batch = (torch.randn(B, C, H, W), torch.randint(0, num_classes, (B,)))
        model.validation_step(batch, 0)

    def test_test_step(self, model: Model):
        batch = (torch.randn(B, C, H, W), torch.randint(0, num_classes, (B,)))
        model.test_step(batch, 0)
