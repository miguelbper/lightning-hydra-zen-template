from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from lightning_hydra_zen_template.sklearn.module import Criterion, Module

N = 5
RNG = np.random.RandomState(42)


@pytest.fixture
def X() -> NDArray:
    return RNG.rand(N, N)


@pytest.fixture
def y() -> NDArray:
    return RNG.rand(N)


@pytest.fixture
def linreg() -> BaseEstimator:
    return LinearRegression(fit_intercept=True)


@pytest.fixture
def metrics() -> list[Criterion]:
    return [mean_squared_error, r2_score]


@pytest.fixture
def model(linreg: BaseEstimator, metrics: list[Criterion]) -> Module:
    return Module(linreg, metrics)


class TestModulePrediction:
    def test_model_call_untrained(self, model: Module, X: NDArray) -> None:
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            model(X)

    def test_model_prediction_matches_base(self, model: Module, X: NDArray, y: NDArray) -> None:
        model.train(X, y)
        y_pred_module = model(X)
        y_pred_base = model.model.predict(X)
        assert np.allclose(y_pred_module, y_pred_base)


class TestModuleTraining:
    def test_model_train_updates_state(self, model: Module, X: NDArray, y: NDArray) -> None:
        assert not model.trained
        model.train(X, y)
        assert model.trained

    def test_model_train_perfect_fit(self, model: Module, X: NDArray, y: NDArray) -> None:
        model.train(X, y)
        y_pred = model(X)
        assert np.allclose(y_pred, y)

    def test_model_retrain_updates_weights(self, model: Module, X: NDArray, y: NDArray) -> None:
        model.train(X, y)
        weights_1 = model.model.coef_.copy()

        X_new = RNG.rand(N, N)
        y_new = RNG.rand(N)
        model.train(X_new, y_new)
        weights_2 = model.model.coef_.copy()

        assert not np.allclose(weights_1, weights_2)

    def test_model_retrain_same_weights(self, model: Module, X: NDArray, y: NDArray) -> None:
        model.train(X, y)
        weights_1 = model.model.coef_.copy()
        model.train(X, y)
        weights_2 = model.model.coef_.copy()
        assert np.allclose(weights_1, weights_2)


class TestModuleEvaluation:
    def test_evaluate_untrained_raises(self, model: Module, X: NDArray, y: NDArray) -> None:
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            model.evaluate(X, y, prefix="test/")

    def test_evaluate_with_multiple_metrics(
        self,
        model: Module,
        X: NDArray,
        y: NDArray,
    ) -> None:
        model.train(X, y)
        results = model.evaluate(X, y, prefix="custom_")
        assert len(results) == 2  # We have MSE and R2
        assert "custom_mean_squared_error" in results
        assert "custom_r2_score" in results

    def test_validate_returns_correct_prefix(
        self,
        model: Module,
        X: NDArray,
        y: NDArray,
    ) -> None:
        model.train(X, y)
        results = model.validate(X, y)
        assert all(k.startswith("val/") for k in results)

    def test_test_returns_correct_prefix(self, model: Module, X: NDArray, y: NDArray) -> None:
        model.train(X, y)
        results = model.test(X, y)
        assert all(k.startswith("test/") for k in results)


class TestModuleSerialization:
    def test_save_load_untrained(self, model: Module, tmp_path: Path) -> None:
        save_path = tmp_path / "untrained_model.pkl"
        model.save(save_path)
        loaded_model = Module.load(save_path)
        assert not loaded_model.trained
        assert len(loaded_model.metrics) == len(model.metrics)

    def test_save_load_trained(self, model: Module, X: NDArray, y: NDArray, tmp_path: Path) -> None:
        model.train(X, y)
        save_path = tmp_path / "trained_model.pkl"
        model.save(save_path)

        loaded_model = Module.load(save_path)
        assert loaded_model.trained
        assert np.allclose(model.model.coef_, loaded_model.model.coef_)
        assert np.allclose(model.model.intercept_, loaded_model.model.intercept_)
        assert np.allclose(model(X), loaded_model(X))

    def test_save_load_preserves_metrics(self, model: Module, tmp_path: Path) -> None:
        save_path = tmp_path / "model_with_metrics.pkl"
        model.save(save_path)
        loaded_model = Module.load(save_path)

        assert len(model.metrics) == len(loaded_model.metrics)
        for m1, m2 in zip(model.metrics, loaded_model.metrics, strict=False):
            assert m1.__name__ == m2.__name__
