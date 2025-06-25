from collections.abc import Callable
from functools import partial
from pathlib import Path

import joblib
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from lightning_hydra_zen_template.utils.types import Metrics, Path_

Criterion = Callable[
    [
        ArrayLike,  # y (target)
        ArrayLike,  # p (preds)
    ],
    float,
]


class SKLearnModule:
    """A wrapper class for scikit-learn models with enhanced functionality.

    This class provides a unified interface for scikit-learn estimators, adding
    support for custom metrics, model persistence, and evaluation methods. It handles
    training state tracking and provides validation/testing capabilities with proper
    metric computation and logging.

    Attributes:
        model (BaseEstimator): The underlying scikit-learn estimator
        metrics (list[Criterion]): List of metric functions for evaluation
        _trained (bool): Internal flag tracking training state
    """

    def __init__(self, model: BaseEstimator, metrics: list[Criterion]):
        """Initialize a SKLearnModule.

        Args:
            model (BaseEstimator): A scikit-learn compatible estimator
            metrics (list[Criterion]): List of metric functions, each taking (target, predictions)
                    and returning a float
        """
        self.model = model
        self.metrics = metrics
        self._trained = False

    @property
    def trained(self) -> bool:
        """Whether the model has been trained.

        Returns:
            bool: True if the model has been trained, False otherwise
        """
        return self._trained

    def __call__(self, X: ArrayLike) -> ArrayLike:
        """Make predictions on input data X.

        Args:
            X (ArrayLike): Input features to make predictions on

        Returns:
            ArrayLike: Model predictions

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def train(self, X: ArrayLike, y: ArrayLike) -> None:
        """Train the model on input features X and target y.

        Args:
            X (ArrayLike): Input features for training
            y (ArrayLike): Target values for training

        Returns:
            None: This method trains the model in-place and does not return values.
        """
        self.model.fit(X, y)
        self._trained = True

    def validate(self, X: ArrayLike, y: ArrayLike) -> Metrics:
        """Validate the model on input features X and target y.

        Args:
            X (ArrayLike): Input features for validation
            y (ArrayLike): Target values for validation

        Returns:
            Metrics: Dictionary with metric names (prefixed with 'val/') as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        return self.evaluate(X, y, prefix="val/")

    def test(self, X: ArrayLike, y: ArrayLike) -> Metrics:
        """Test the model on input features X and target y.

        Args:
            X (ArrayLike): Input features for testing
            y (ArrayLike): Target values for testing

        Returns:
            Metrics: Dictionary with metric names (prefixed with 'test/') as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        return self.evaluate(X, y, prefix="test/")

    def evaluate(self, X: ArrayLike, y: ArrayLike, prefix: str) -> Metrics:
        """Evaluate the model on input features X and target y.

        Computes predictions and evaluates all metrics, returning results with
        the specified prefix. Metric names are extracted from function names,
        handling both regular functions and partial functions.

        Args:
            X (ArrayLike): Input features for evaluation
            y (ArrayLike): Target values for evaluation
            prefix (str): Prefix for metric names (e.g., 'val/' or 'test/')

        Returns:
            Metrics: Dictionary with metric names (prefixed) as keys and their values

        Raises:
            RuntimeError: If the model has not been trained yet
        """
        p = self(X)
        results = {}
        for metric in self.metrics:
            metric_name = metric.func.__name__ if isinstance(metric, partial) else metric.__name__
            metric_value = metric(y, p)
            results[f"{prefix}{metric_name}"] = metric_value
        return results

    def save(self, path: Path_) -> None:
        """Save the entire SKLearnModule object to the specified path.

        Uses joblib to serialize and save the complete module object, including
        the trained model, metrics, and training state.

        Args:
            path (Path_): Path where to save the model

        Returns:
            None: This method saves the model to disk but does not return values.

        Raises:
            OSError: If the file cannot be written
            ValueError: If the object cannot be serialized
        """
        path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path_) -> "SKLearnModule":
        """Load a SKLearnModule object from the specified path.

        Uses joblib to deserialize and load a previously saved module object,
        including the trained model, metrics, and training state.

        Args:
            path (Path_): Path from where to load the model

        Returns:
            SKLearnModule: The loaded module object

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be deserialized
        """
        path = Path(path)
        return joblib.load(path)
