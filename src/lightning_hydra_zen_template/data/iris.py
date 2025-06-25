from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lightning_hydra_zen_template.scikit_learn.datamodule import DataModule
from lightning_hydra_zen_template.utils.types import Data


class IrisDataModule(DataModule):
    """DataModule for the Iris dataset.

    This module loads the Iris dataset from scikit-learn and provides train, validation,
    and test splits. Features are standardized using StandardScaler.

    Attributes:
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        standardize: Whether to standardize features
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int | None = 42,
        standardize: bool = True,
    ) -> None:
        """Initialize the Iris DataModule.

        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            val_size: Fraction of training data to use for validation (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            standardize: Whether to standardize features (default: True)
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.standardize = standardize

        iris = load_iris()
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        X, y = iris.data, iris.target

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_train_val,
        )

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        self._X_test = X_test
        self._y_test = y_test

    def train_set(self) -> Data:
        """Return the training data.

        Returns:
            Data: Tuple of (features, targets) for training
        """
        return self._X_train, self._y_train

    def validation_set(self) -> Data:
        """Return the validation data.

        Returns:
            Data: Tuple of (features, targets) for validation
        """
        return self._X_val, self._y_val

    def test_set(self) -> Data:
        """Return the test data.

        Returns:
            Data: Tuple of (features, targets) for testing
        """
        return self._X_test, self._y_test
