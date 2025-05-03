import os

from hydra_zen import store

from lightning_hydra_zen_template.configs.groups.paths import root_dir
from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.data.mnist import MNISTDataModule

MNISTDataModuleCfg = fbuilds(
    MNISTDataModule,
    data_dir=os.path.join(root_dir, "data"),
    zen_wrappers=log_instantiation,
)

store(MNISTDataModuleCfg, group="data", name="mnist")
