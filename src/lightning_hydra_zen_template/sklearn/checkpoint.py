import os
from typing import Literal

from torch import Tensor, tensor

from lightning_hydra_zen_template.sklearn.datamodule import DataModule
from lightning_hydra_zen_template.sklearn.module import Module
from lightning_hydra_zen_template.utils.types import Metrics


class SKLearnCheckpoint:
    def __init__(
        self,
        dirpath: str,
        monitor: str,
        mode: Literal["min", "max"],
    ) -> None:
        self.dirpath: str = dirpath
        self.monitor: str = monitor
        self.mode: Literal["min", "max"] = mode
        self.model_path: str = os.path.join(self.dirpath, "ckpt.pkl")
        self.val_metrics: Metrics = {}
        self.test_metrics: Metrics = {}

    def save(self, model: Module, datamodule: DataModule) -> None:
        model.save(self.model_path)

        X, y = datamodule.validation_set()
        self.val_metrics: Metrics = model.validate(X, y)

        X, y = datamodule.test_set()
        self.test_metrics: Metrics = model.test(X, y)

    @property
    def best_model_path(self) -> str:
        return self.model_path

    @property
    def best_model_score(self) -> Tensor | None:
        metrics: Metrics = {**self.val_metrics, **self.test_metrics}
        score: float | None = metrics.get(self.monitor, None)
        return tensor(score) if score is not None else None
