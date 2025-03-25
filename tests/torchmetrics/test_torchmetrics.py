"""Tests to understand how torchmetrics works.

Based on the discussion in https://github.com/Lightning-AI/torchmetrics/issues/1717,
the recommended averaging strategies for different metrics are:

- Accuracy: average="micro"
- Precision: average="macro"
- Recall: average="macro"
- F1Score: average="macro"
"""

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics.functional.classification import multiclass_accuracy

C = 3
N = 10

p_shape = (N, C)
t_shape = (N,)


def accuracy(p: Tensor, t: Tensor) -> Tensor:
    predicted_classes: Tensor = torch.argmax(p, dim=1)
    correct: Tensor = predicted_classes == t
    return torch.mean(correct.float())


@pytest.fixture
def t_zeros() -> Tensor:
    return torch.zeros(t_shape, dtype=torch.int32)


@pytest.fixture
def p_zeros() -> Tensor:
    return torch.zeros(p_shape, dtype=torch.int32) + torch.tensor([1, 0, 0], dtype=torch.int32)


@pytest.fixture
def p_ones() -> Tensor:
    return torch.zeros(p_shape, dtype=torch.int32) + torch.tensor([0, 1, 0], dtype=torch.int32)


@pytest.fixture(params=range(10))
def t_rand(request: FixtureRequest) -> Tensor:
    torch.manual_seed(request.param)
    return torch.randint(0, C, t_shape, dtype=torch.int32)


@pytest.fixture(params=range(10))
def p_rand(request: FixtureRequest) -> Tensor:
    torch.manual_seed(request.param)
    return torch.rand(p_shape, dtype=torch.float32)


@pytest.fixture
def accuracy_metric() -> Metric:
    return Accuracy(task="multiclass", num_classes=C, average="micro")


@pytest.fixture
def metrics_grouped() -> MetricCollection:
    return MetricCollection(
        [
            Accuracy(task="multiclass", num_classes=C, average="micro"),
            Precision(task="multiclass", num_classes=C, average="macro"),
            Recall(task="multiclass", num_classes=C, average="macro"),
            F1Score(task="multiclass", num_classes=C, average="macro"),
        ],
        compute_groups=True,
    )


@pytest.fixture
def metrics_ungrouped() -> MetricCollection:
    return MetricCollection(
        [
            Accuracy(task="multiclass", num_classes=C, average="micro"),
            Precision(task="multiclass", num_classes=C, average="macro"),
            Recall(task="multiclass", num_classes=C, average="macro"),
            F1Score(task="multiclass", num_classes=C, average="macro"),
        ],
        compute_groups=False,
    )


class TestAccuracy:
    def test_all_correct(self, t_zeros: Tensor, p_zeros: Tensor) -> None:
        assert accuracy(p_zeros, t_zeros) == 1.0

    def test_all_incorrect(self, t_zeros: Tensor, p_ones: Tensor) -> None:
        assert accuracy(p_ones, t_zeros) == 0.0

    def test_functional(self, t_rand: Tensor, p_rand: Tensor) -> None:
        a0 = accuracy(p_rand, t_rand)
        a1 = multiclass_accuracy(p_rand, t_rand, num_classes=C, average="micro")
        assert a0 == a1

    def test_class(self, t_rand: Tensor, p_rand: Tensor, accuracy_metric: Metric) -> None:
        a0 = accuracy(p_rand, t_rand)
        a1 = accuracy_metric(p_rand, t_rand)
        assert a0 == a1

    def test_metric_collection(self, t_rand: Tensor, p_rand: Tensor, metrics_grouped: MetricCollection) -> None:
        a0 = accuracy(p_rand, t_rand)
        a1 = metrics_grouped(p_rand, t_rand)["MulticlassAccuracy"]
        assert a0 == a1

    def test_compute_groups(
        self,
        t_rand: Tensor,
        p_rand: Tensor,
        metrics_grouped: MetricCollection,
        metrics_ungrouped: MetricCollection,
    ) -> None:
        a0 = metrics_grouped(p_rand, t_rand)
        a1 = metrics_ungrouped(p_rand, t_rand)
        for k in a0:
            assert a0[k] == a1[k]
