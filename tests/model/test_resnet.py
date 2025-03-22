import torch
from torch import nn

from src.model.resnet import ResNet


def test_instantiation():
    resnet = ResNet()
    assert resnet is not None
    assert isinstance(resnet, nn.Module)


def test_forward_rgb():
    resnet = ResNet()
    x = torch.randn(1, 3, 224, 224)
    y = resnet(x)
    assert y.shape == (1, 10)
