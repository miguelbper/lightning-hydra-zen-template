from pathlib import Path

from numpy.typing import NDArray
from torch import Tensor

Metrics = dict[str, float]
Path_ = str | Path

# torch: X, y
Batch = tuple[
    Tensor,  # X = features
    Tensor,  # y = target
]

# numpy: X, y
Data = tuple[
    NDArray,  # X = features
    NDArray,  # y = target
]
