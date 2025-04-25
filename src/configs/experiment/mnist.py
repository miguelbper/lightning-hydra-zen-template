from hydra_zen import builds, make_config
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)

from src.configs.config import Config, data_dir, experiment_store, log_instantiation
from src.datamodule.mnist import MNISTDataModule
from src.model.model import Model
from src.model.resnet import ResNet

NUM_CLASSES = 10


DataModuleCfg = builds(
    MNISTDataModule,
    data_dir=data_dir,
    zen_wrappers=log_instantiation,
)


ModelCfg = builds(
    Model,
    net=builds(
        ResNet,
        num_classes=NUM_CLASSES,
    ),
    loss_fn=builds(
        nn.CrossEntropyLoss,
    ),
    optimizer=builds(
        Adam,
        zen_partial=True,
    ),
    scheduler=builds(
        ReduceLROnPlateau,
        mode="min",
        factor=0.1,
        patience=10,
        zen_partial=True,
    ),
    metric_collection=builds(
        MetricCollection,
        metrics=[
            builds(
                Accuracy,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="micro",
            ),
            builds(
                F1Score,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            builds(
                Precision,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            builds(
                Recall,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
        ],
    ),
    zen_wrappers=log_instantiation,
)

MNISTCfg = make_config(
    task_name="mnist",
    monitor="val/MulticlassAccuracy",
    mode="max",
    datamodule=DataModuleCfg,
    model=ModelCfg,
    bases=(Config,),
)

experiment_store(MNISTCfg, name="mnist")
