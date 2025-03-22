import torch
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2

Input = torch.Tensor
Target = torch.Tensor
Batch = tuple[Input, Target]

MNIST_NUM_TRAIN_EXAMPLES = 60000
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_val_examples: int = 5000,
    ) -> None:
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
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.mnist_train, self.mnist_val = random_split(
                dataset=MNIST(self.data_dir, train=True, transform=self.transform),
                lengths=[self.num_train_examples, self.num_val_examples],
                generator=Generator().manual_seed(42),
            )
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
