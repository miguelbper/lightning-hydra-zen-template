from pathlib import Path

import torch
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2

Input = torch.Tensor
Target = torch.Tensor
Batch = tuple[Input, Target]

MNIST_NUM_TRAIN_EXAMPLES = 60000
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTDataModule(LightningDataModule):
    """A PyTorch Lightning DataModule for the MNIST dataset.

    This class handles downloading, preprocessing, and loading of the MNIST dataset.
    It splits the training data into training and validation sets, and provides
    separate dataloaders for training, validation, and testing.

    Attributes:
        data_dir (str | Path): Directory where MNIST dataset is stored.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to pin memory in CPU for faster GPU transfer.
        num_val_examples (int): Number of examples to use for validation.
        num_train_examples (int): Number of examples to use for training.
        transform (v2.Compose): Image transformations to apply to the data.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_val_examples: int = 5000,
    ) -> None:
        """Initialize the MNIST DataModule.

        Args:
            data_dir (str | Path): Directory where MNIST dataset is stored.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory in CPU. Defaults to False.
            num_val_examples (int, optional): Number of validation examples. Defaults to 5000.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
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
