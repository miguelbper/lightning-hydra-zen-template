import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig


class TestModel:
    def test_instantiation(self, cfg: DictConfig):
        HydraConfig().set_config(cfg)

        model = instantiate(cfg.model)
        assert model is not None
        assert isinstance(model, LightningModule)

    def test_configure_optimizers(self, cfg: DictConfig):
        HydraConfig().set_config(cfg)

        model = instantiate(cfg.model)
        optim_cfg = model.configure_optimizers()
        assert optim_cfg is not None
        assert "optimizer" in optim_cfg

    def test_training_step(self, cfg: DictConfig):
        HydraConfig().set_config(cfg)

        model = instantiate(cfg.model)
        batch = (torch.randn(32, 3, 224, 224), torch.randint(0, 10, (32,)))
        loss = model.training_step(batch, 0)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_validation_step(self, cfg: DictConfig):
        pass

    def test_test_step(self, cfg: DictConfig):
        pass
