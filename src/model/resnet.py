import torch
from torch import nn


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()  # type: ignore
        self.num_classes: int = num_classes
        self.model: nn.Module = torch.hub.load(  # nosec B614  # type: ignore
            "pytorch/vision:v0.10.0",
            "resnet18",
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
