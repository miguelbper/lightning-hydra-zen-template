from lightning import LightningDataModule


class MNISTDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./data",
            batch_size: int = 32,
            num_workers: int = 0
            ) -> None:
        super().__init__()
        self.save_hyperparameters()
