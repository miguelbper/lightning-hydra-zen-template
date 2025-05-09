import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet(nn.Module):
    """A wrapper class for ResNet18 model with customizable output layer.

    This class provides a pre-trained ResNet18 model from torchvision with a
    customizable final fully connected layer for transfer learning tasks.

    Attributes:
        model (nn.Module): The underlying ResNet18 model with modified final layer.
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the ResNet model.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 10.
        """
        super().__init__()
        self.model: nn.Module = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C).
        """
        return self.model(x)
