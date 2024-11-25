from lightning import LightningDataModule
from torch.utils.data import DataLoader


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    # TODO: add specific type hint for DataLoader
    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass
