import pytest
import torch
from hydra_zen import instantiate
from lightning import LightningModule

from lightning_hydra_zen_template.configs.groups.model import ModelCfg
from lightning_hydra_zen_template.model.model import Model

B, C, H, W = 2, 3, 32, 32
num_classes = 10


@pytest.fixture(params=[1, 2])
def batch_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def model() -> Model:
    return instantiate(ModelCfg)


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

    def test_validation_step(self, model: Model, batch_size: int):
        model.eval()
        with torch.no_grad():
            batch = (torch.randn(batch_size, C, H, W), torch.randint(0, num_classes, (batch_size,)))
            model.validation_step(batch, 0)

    def test_test_step(self, model: Model, batch_size: int):
        model.eval()
        with torch.no_grad():
            batch = (torch.randn(batch_size, C, H, W), torch.randint(0, num_classes, (batch_size,)))
            model.test_step(batch, 0)
