import torch
from torch import nn

from src.model.resnet import ResNet

B, C, H, W = 32, 3, 32, 32
num_classes = 10


def test_instantiation():
    resnet = ResNet()
    assert resnet is not None
    assert isinstance(resnet, nn.Module)


def test_forward_rgb():
    resnet = ResNet()
    x = torch.randn(B, C, H, W)
    y = resnet(x)
    assert y.shape == (B, num_classes)
