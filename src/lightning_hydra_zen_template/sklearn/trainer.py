from collections.abc import Callable
from typing import Literal

from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT
from numpy.typing import ArrayLike
from rich.console import Console
from rich.table import Table

from lightning_hydra_zen_template.sklearn.checkpoint import SKLearnCheckpoint
from lightning_hydra_zen_template.sklearn.datamodule import DataModule
from lightning_hydra_zen_template.sklearn.module import Module
from lightning_hydra_zen_template.utils.types import Metrics, Path_


class SKLearnTrainer:
    """A trainer class for scikit-learn models with Numerai-specific
    evaluation.

    This class provides training, validation, and testing capabilities
    for scikit-learn models wrapped in the Module class. It handles
    model persistence, evaluation on different data splits, and pretty
    printing of results using Rich tables.

    The trainer supports checkpointing for model persistence and can
    load models from saved checkpoints for evaluation.
    """

    def __init__(self, checkpoint_callback: SKLearnCheckpoint) -> None:
        self.checkpoint_callback = checkpoint_callback

    def fit(self, model: Module, datamodule: DataModule, ckpt_path: Path_ | None = None) -> None:
        """Train a model using the provided datamodule.

        Args:
            model: The model to train
            datamodule: DataModule containing the training data
            ckpt_path: Optional path to save the trained model checkpoint
        """
        X, y = datamodule.train_set()
        model.train(X, y)
        self.checkpoint_callback.save(model=model, datamodule=datamodule)

    def validate(
        self, model: Module | None, datamodule: DataModule, ckpt_path: Path_ | None = None
    ) -> _EVALUATE_OUTPUT:
        """Evaluate the model on validation data.

        Args:
            model: The model to validate, can be None if ckpt_path is provided
            datamodule: DataModule containing the validation data
            ckpt_path: Optional path to load model from, required if model is None

        Returns:
            Metrics: Dictionary of validation metrics

        Raises:
            ValueError: If neither model nor ckpt_path is provided, or if model is not trained
        """
        return self.evaluate(model, datamodule, ckpt_path, "validation")

    def test(self, model: Module | None, datamodule: DataModule, ckpt_path: Path_ | None = None) -> _EVALUATE_OUTPUT:
        """Evaluate the model on test data.

        Args:
            model: The model to test, can be None if ckpt_path is provided
            datamodule: DataModule containing the test data
            ckpt_path: Optional path to load model from, required if model is None

        Returns:
            Metrics: Dictionary of test metrics

        Raises:
            ValueError: If neither model nor ckpt_path is provided, or if model is not trained
        """
        return self.evaluate(model, datamodule, ckpt_path, "test")

    def evaluate(
        self,
        model: Module | None,
        datamodule: DataModule,
        ckpt_path: Path_ | None,
        split: Literal["validation", "test"],
    ) -> _EVALUATE_OUTPUT:
        """Common evaluation logic for validation and test.

        Args:
            model: Model to evaluate, can be None if ckpt_path is provided
            datamodule: DataModule containing the data
            ckpt_path: Optional path to load model from, required if model is None
            split: Which split to evaluate ('validation' or 'test')

        Returns:
            Metrics: Dictionary of evaluation metrics

        Raises:
            ValueError: If neither model nor ckpt_path is provided, or if model is not trained
        """
        checkpoint_attr: str = "val_metrics" if split == "validation" else "test_metrics"
        metrics: Metrics = getattr(self.checkpoint_callback, checkpoint_attr)

        if not metrics:
            if model is None and ckpt_path is None:
                raise ValueError("Either model or ckpt_path must be provided")
            if ckpt_path:
                model = Module.load(ckpt_path)
            if not model.trained:
                raise ValueError("Model must be trained before evaluation")

            datamodule_func_name: str = "validation_set" if split == "validation" else "test_set"
            datamodule_func: Callable[[], tuple[ArrayLike, ArrayLike]] = getattr(datamodule, datamodule_func_name)
            split_prefix: str = "val/" if split == "validation" else "test/"
            X, y = datamodule_func()
            metrics: Metrics = model.evaluate(X, y, split_prefix)

        print_metrics(metrics, split.capitalize())
        return [metrics]


def print_metrics(metrics: Metrics, prefix: str) -> None:
    """Pretty print metrics in a table format using Rich.

    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix to use in the title (e.g., 'Validation' or 'Test')
    """
    console = Console()
    table = Table()
    table.add_column(f"{prefix} metric", style="cyan")
    table.add_column("Value", style="magenta")

    for name, value in metrics.items():
        table.add_row(name, f"{value:.16f}")

    console.print(table)
