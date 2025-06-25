import os

import torch
from einops import asnumpy, rearrange
from lightning import LightningDataModule
from lightning.fabric.utilities.data import suggested_max_num_workers
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from lightning_hydra_zen_template.configs.utils.paths import data_dir
from lightning_hydra_zen_template.scikit_learn.datamodule import DataModule
from lightning_hydra_zen_template.utils.types import Batch, Data, Path_

MNIST_NUM_TRAIN_EXAMPLES: int = 60000
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081

raw_data_dir: str = os.path.join(data_dir, "raw")


class MNISTDataModule(LightningDataModule, DataModule):
    """A PyTorch Lightning DataModule for the MNIST dataset.

    This class handles downloading, preprocessing, and loading of the MNIST dataset.
    It splits the training data into training and validation sets, and provides
    separate dataloaders for training, validation, and testing.

    Attributes:
        data_dir (Path_): Directory where MNIST dataset is stored.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to pin memory in CPU for faster GPU transfer.
        num_val_examples (int): Number of examples to use for validation.
        num_train_examples (int): Number of examples to use for training.
        transform (v2.Compose): Image transformations to apply to the data.
    """

    def __init__(
        self,
        data_dir: Path_ = raw_data_dir,
        num_val_examples: int = 5000,
        batch_size: int = 32,
        num_workers: int | None = None,
        num_devices: int | None = None,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the MNIST DataModule.

        Args:
            data_dir (Path_): Directory where MNIST dataset is stored.
            num_val_examples (int, optional): Number of validation examples. Defaults to 5000.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0. If None, the number of
                workers is automatically computed based on the number of devices.
            num_devices (int, optional): Number of devices in trainer. Only relevant if num_workers is not None, to do
                automatic computation of num_workers.
            pin_memory (bool, optional): Whether to pin memory in CPU. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_devices = num_devices or 1
        self.num_workers = suggested_max_num_workers(self.num_devices) if num_workers is None else num_workers
        self.pin_memory = pin_memory
        self.num_val_examples = num_val_examples
        self.num_train_examples = MNIST_NUM_TRAIN_EXAMPLES - num_val_examples
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.RGB(),
                v2.Pad(2),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[MNIST_MEAN for _ in range(3)], std=[MNIST_STD for _ in range(3)]),
            ]
        )

    def prepare_data(self) -> None:
        """Download MNIST dataset if it doesn't exist."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Set up the datasets for training, validation, or testing.

        Args:
            stage (str): Either 'fit' (training) or 'test'.
        """
        if stage == "fit":
            dataset: Dataset[Batch] = MNIST(self.data_dir, train=True, transform=self.transform)
            lengths: list[int] = [self.num_train_examples, self.num_val_examples]
            generator: Generator = Generator().manual_seed(42)
            splits: list[Subset[Batch]] = random_split(dataset=dataset, lengths=lengths, generator=generator)
            self.mnist_train: Dataset[Batch] = splits[0]
            self.mnist_val: Dataset[Batch] = splits[1]
        if stage == "test":
            self.mnist_test: Dataset[Batch] = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader[Batch]:
        """Create and return the training dataloader.

        Returns:
            DataLoader[Batch]: Dataloader for training data with shuffling enabled.
        """
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader[Batch]: Dataloader for validation data with shuffling disabled.
        """
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Batch]:
        """Create and return the test dataloader.

        Returns:
            DataLoader[Batch]: Dataloader for test data with shuffling disabled.
        """
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def train_dataset(self) -> Data:
        if not hasattr(self, "mnist_train"):
            self.setup(stage="fit")
        return self.get_dataset(self.mnist_train)

    def val_dataset(self) -> Data:
        if not hasattr(self, "mnist_val"):
            self.setup(stage="fit")
        return self.get_dataset(self.mnist_val)

    def test_dataset(self) -> Data:
        if not hasattr(self, "mnist_test"):
            self.setup(stage="test")
        return self.get_dataset(self.mnist_test)

    def get_dataset(self, dataset: Dataset[Batch]) -> Data:
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=self.num_workers)
        images: list[Tensor] = []
        labels: list[Tensor] = []
        for image, label in loader:
            images.append(image)
            labels.append(label)
        images: Tensor = torch.cat(images, dim=0)
        labels: Tensor = torch.cat(labels, dim=0)

        grayscale: Tensor = images[:, 0]  # n c h w -> n h w
        X: Tensor = rearrange(grayscale, "n h w -> n (h w)")
        y: Tensor = labels
        return asnumpy(X), asnumpy(y)
