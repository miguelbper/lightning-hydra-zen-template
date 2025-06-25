from hydra_zen import store

from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.data.mnist import MNISTDataModule

MNISTDataModuleCfg = fbuilds(
    MNISTDataModule,
    zen_wrappers=log_instantiation,
)


data_store = store(group="data")
data_store(MNISTDataModuleCfg, name="mnist")
