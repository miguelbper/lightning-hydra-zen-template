import pytest
from rootutils import find_root

from lightning_hydra_zen_template.data.mnist import MNISTDataModule

ROOT_DIR = find_root()
DATA_DIR = ROOT_DIR / "data" / "raw"
C, H, W = 3, 32, 32


@pytest.fixture(params=[1, 2])
def batch_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def datamodule(batch_size: int) -> MNISTDataModule:
    dm = MNISTDataModule(data_dir=DATA_DIR, batch_size=batch_size)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")
    return dm


class TestMNISTDataModule:
    def test_train_dataloader(self, datamodule: MNISTDataModule, batch_size: int):
        train_dataloader = datamodule.train_dataloader()
        batch = next(iter(train_dataloader))
        images, labels = batch
        assert images.shape == (batch_size, C, H, W)
        assert labels.shape == (batch_size,)

    def test_val_dataloader(self, datamodule: MNISTDataModule, batch_size: int):
        val_dataloader = datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        images, labels = batch
        assert images.shape == (batch_size, C, H, W)
        assert labels.shape == (batch_size,)

    def test_test_dataloader(self, datamodule: MNISTDataModule, batch_size: int):
        test_dataloader = datamodule.test_dataloader()
        batch = next(iter(test_dataloader))
        images, labels = batch
        assert images.shape == (batch_size, C, H, W)
        assert labels.shape == (batch_size,)
