import torch
from torch import nn

from lightning_hydra_zen_template.model.components.resnet import ResNet

B, C, H, W = 32, 3, 32, 32
num_classes = 10


class TestResNet:
    def test_instantiation(self):
        resnet = ResNet(num_classes=num_classes)
        assert resnet is not None
        assert isinstance(resnet, nn.Module)

    def test_forward_rgb(self):
        resnet = ResNet(num_classes=num_classes)
        x = torch.randn(B, C, H, W)
        y = resnet(x)
        assert y.shape == (B, num_classes)
