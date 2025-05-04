from hydra_zen import store
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from lightning_hydra_zen_template.configs.utils.utils import fbuilds, log_instantiation
from lightning_hydra_zen_template.model.components.resnet import ResNet
from lightning_hydra_zen_template.model.model import Model

NUM_CLASSES = 10

ModelCfg = fbuilds(
    Model,
    net=fbuilds(
        ResNet,
        num_classes=NUM_CLASSES,
    ),
    loss_fn=fbuilds(
        nn.CrossEntropyLoss,
    ),
    optimizer=fbuilds(
        Adam,
        lr=0.001,
        zen_partial=True,
    ),
    scheduler=fbuilds(
        ReduceLROnPlateau,
        mode="min",
        factor=0.1,
        patience=10,
        zen_partial=True,
    ),
    metric_collection=fbuilds(
        MetricCollection,
        metrics=[
            fbuilds(
                Accuracy,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="micro",
            ),
            fbuilds(
                F1Score,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            fbuilds(
                Precision,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            fbuilds(
                Recall,
                task="multiclass",
                num_classes=NUM_CLASSES,
                average="macro",
            ),
        ],
    ),
    zen_wrappers=log_instantiation,
)

store(ModelCfg, group="model", name="mnist")
