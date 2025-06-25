from abc import ABC, abstractmethod

from lightning_hydra_zen_template.utils.types import Data


class DataModule(ABC):
    """Abstract base class for data modules in the sklearn framework.

    This class defines the interface for data modules that provide access to
    training, validation, and test datasets. Each dataset is returned as a
    tuple containing era indices, features, targets, and meta model predictions.

    The Data type is defined as:
        Data = tuple[NDArray, NDArray]
        where the elements are: (features, targets)
    """

    @abstractmethod
    def train_dataset(self) -> Data:
        """Get the training dataset.

        Returns:
            Data: A tuple containing:
                - features: Input features for training
                - target: Target values for training
        """
        pass  # pragma: no cover

    @abstractmethod
    def val_dataset(self) -> Data:
        """Get the validation dataset.

        Returns:
            Data: A tuple containing:
                - features: Input features for validation
                - target: Target values for validation
        """
        pass  # pragma: no cover

    @abstractmethod
    def test_dataset(self) -> Data:
        """Get the test dataset.

        Returns:
            Data: A tuple containing:
                - features: Input features for testing
                - target: Target values for testing
        """
        pass  # pragma: no cover
