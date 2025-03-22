import rootutils
from lightning import LightningDataModule

from src.datamodule.mnist import MNISTDataModule

ROOT_DIR = rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=False)

DATA_DIR = ROOT_DIR / "data/raw"


def test_instantiation():
    datamodule = MNISTDataModule(data_dir=DATA_DIR)
    assert datamodule is not None
    assert isinstance(datamodule, LightningDataModule)


def test_shape():
    datamodule = MNISTDataModule(data_dir=DATA_DIR)
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    images, labels = batch
    assert images.shape == (32, 3, 224, 224)
    assert labels.shape == (32,)
