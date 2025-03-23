import pytest
import rootutils

from src.datamodule.mnist import MNISTDataModule

ROOT_DIR = rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=False)
DATA_DIR = ROOT_DIR / "data" / "raw"
B, C, H, W = 32, 3, 32, 32


@pytest.fixture
def datamodule() -> MNISTDataModule:
    dm = MNISTDataModule(data_dir=DATA_DIR)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")
    return dm


class TestMNISTDataModule:
    def test_train_dataloader(self, datamodule: MNISTDataModule):
        train_dataloader = datamodule.train_dataloader()
        batch = next(iter(train_dataloader))
        images, labels = batch
        assert images.shape == (B, C, H, W)
        assert labels.shape == (B,)

    def test_val_dataloader(self, datamodule: MNISTDataModule):
        val_dataloader = datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        images, labels = batch
        assert images.shape == (B, C, H, W)
        assert labels.shape == (B,)

    def test_test_dataloader(self, datamodule: MNISTDataModule):
        test_dataloader = datamodule.test_dataloader()
        batch = next(iter(test_dataloader))
        images, labels = batch
        assert images.shape == (B, C, H, W)
        assert labels.shape == (B,)
