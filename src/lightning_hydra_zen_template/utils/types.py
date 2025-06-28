from pathlib import Path

from torch import Tensor

Metrics = dict[str, float]
Path_ = str | Path

# torch: X, y
Batch = tuple[
    Tensor,  # X = features
    Tensor,  # y = target
]
