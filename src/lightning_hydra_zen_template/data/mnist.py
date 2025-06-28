import os

import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.data import suggested_max_num_workers
from torch import Generator
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from lightning_hydra_zen_template.configs.utils.paths import data_dir
from lightning_hydra_zen_template.utils.types import Batch, Path_

MNIST_NUM_TRAIN_EXAMPLES: int = 60000
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081

raw_data_dir: str = os.path.join(data_dir, "raw")


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path_ = raw_data_dir,
        num_val_examples: int = 5000,
        batch_size: int = 1024,
        num_devices: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the MNIST DataModule.

        Args:
            data_dir (Path_): Directory where MNIST dataset is stored.
            num_val_examples (int, optional): Number of validation examples. Defaults to 5000.
            batch_size (int, optional): Number of samples per batch. Defaults to 1024.
            num_devices (int, optional): Number of devices in trainer. Only relevant if num_workers is not None, to do
                automatic computation of num_workers.
            num_workers (int, optional): Number of subprocesses for data loading. If None, the number of
                workers is automatically computed based on the number of devices.
            pin_memory (bool, optional): Whether to pin memory in CPU. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir: Path_ = data_dir
        self.num_val_examples: int = num_val_examples
        self.batch_size: int = batch_size
        self.num_devices: int = num_devices or 1
        self.num_workers: int = suggested_max_num_workers(self.num_devices) if num_workers is None else num_workers
        self.pin_memory: bool = pin_memory

        self.num_train_examples: int = MNIST_NUM_TRAIN_EXAMPLES - num_val_examples
        self.transform: v2.Compose = v2.Compose(
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
            DataLoader[Batch]: Dataloader for training data.
        """
        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader[Batch]: Dataloader for validation data.
        """
        return DataLoader(
            dataset=self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Batch]:
        """Create and return the test dataloader.

        Returns:
            DataLoader[Batch]: Dataloader for test data.
        """
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
