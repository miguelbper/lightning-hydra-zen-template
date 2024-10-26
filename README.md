# Deep Learning Template
[![python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra-89b8cd?logo=meta)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/-Ruff-261230?logo=ruff&logoColor=261230&labelColor=d7ff64)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/-uv-261230?logo=astral&logoColor=261230&labelColor=de5fe9)](https://github.com/astral-sh/uv)
[![license](https://img.shields.io/badge/license-MIT-green.svg?labelColor=gray)](https://github.com/miguelbper/deep-learning-template/blob/main/LICENSE)
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->
<!-- [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) -->
<!-- add automatic tests passing -->
<!-- add automatic code coverage -->



Template for deep learning projects, using
- [PyTorch](https://github.com/pytorch/pytorch) - Neural networks
- [Lightning](https://github.com/Lightning-AI/pytorch-lightning) - PyTorch training loop abstraction
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) - Metrics for Lightning
- [TensorBoard](https://github.com/tensorflow/tensorboard) - Training monitoring
- [Optuna](https://github.com/optuna/optuna) - Hyperparameter optimization
- [MLflow](https://github.com/mlflow/mlflow) - Experiment tracking
- [Hydra](https://github.com/facebookresearch/hydra) - Configuration files
- [Ruff](https://github.com/astral-sh/ruff) - Linting and formatting
- [uv](https://github.com/astral-sh/uv) - Dependency management


## Setup
1. Fork the repo or copy the code to the working directory of the new project. Move to the working directory.

2. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install python with uv:
```bash
uv python install
```

4. Install the virtual environment (this will create a venv folder in the working directory):
```bash
uv sync
```

5. Run the unit tests:
```bash
uv run pytest
```


## References

Based on the very nice [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
