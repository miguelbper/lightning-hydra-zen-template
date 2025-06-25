from abc import ABC, abstractmethod

from lightning_hydra_zen_template.utils.types import Data


class SKLearnDataModule(ABC):
    """Abstract base class for data modules in the sklearn framework.

    This class defines the interface for data modules that provide
    access to training, validation, and test datasets. Each dataset is
    returned as a tuple containing features and targets.
    """

    @abstractmethod
    def train_dataset(self) -> Data:
        """Get the training dataset.

        Returns:
            Data: A tuple containing:
                - features: Input features
                - targets: Target values
        """
        pass  # pragma: no cover

    @abstractmethod
    def val_dataset(self) -> Data:
        """Get the validation dataset.

        Returns:
            Data: A tuple containing:
                - features: Input features
                - targets: Target values
        """
        pass  # pragma: no cover

    @abstractmethod
    def test_dataset(self) -> Data:
        """Get the test dataset.

        Returns:
            Data: A tuple containing:
                - features: Input features
                - targets: Target values
        """
        pass  # pragma: no cover
