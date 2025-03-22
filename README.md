# Deep Learning Template
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![PyTorch Lightning](https://img.shields.io/badge/-Lightning-7e4fff?logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra-89b8cd?logo=meta)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/Ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/uv-de5fe9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![license](https://img.shields.io/badge/license-MIT-green.svg?labelColor=gray)](https://github.com/miguelbper/deep-learning-template/blob/main/LICENSE)
<!-- TODO: add automatic tests passing -->
<!-- TODO: add automatic code coverage -->



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


## Directory structure
```
├── configs                 <- Configuration files for Hydra, containing model, training, and experiment settings
│   └── ...
│
├── data                    <- Directory for datasets
│   ├── interim             <- Intermediate results of dataset processing
│   ├── processed           <- Datasets ready to be used by the modelling scripts
│   └── raw                 <- Datasets as obtained from the source
│
├── logs                    <- Training logs, metrics, checkpoints, and experiment tracking data
├── notebooks               <- Jupyter notebooks for experimentation
├── scripts                 <- Shell scripts
│
├── src                     <- Source code for the project
│   ├── datamodule          <- Lightning DataModules for handling datasets
│   ├── model               <- Neural network model definitions and Lightning Modules
│   ├── ...
│   ├── test.py             <- Testing / evaluation script
│   └── train.py            <- Training script
│
├── tests                   <- Automated tests
│   └── ...
│
├── .envrc-example          <- Environment variables, automatically loaded with direnv
├── .gitignore              <- Specifies which files Git should ignore
├── .pre-commit-config.yaml <- Git pre-commit hooks
├── .python-version         <- Python version that should be installed
├── justfile                <- Project commands
├── LICENSE                 <- MIT License file
├── pyproject.toml          <- Project configuration file with dependencies and tool settings
├── README.md               <- The top-level README for developers using this project
└── uv.lock                 <- The requirements file for reproducing the environment
```

## Setup
1. Click the green "Use this template" button on GitHub to start a new project.

2. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install the virtual environment (this will create a `.venv` folder in the working directory):
```bash
uv sync
```
<!-- TODO: review setup instructions, taking into account that they should be self documenting in just -->

## Acknowledgements
As a reference for this template, I used the following very nice projects:
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [drivendataorg/cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science)
<!-- TODO: at some point, may add what motivated me to do these changes, relative to the repos that already exist
- Borrow ideas from both
- Better dependency management with uv
- Better linting and formatting with Ruff
- Better generic LightningModule, not adapted to dataset at hand
- Add a justfile
- Learning exercise
-->

<!-- ###########################################################################
TODOS
- TODO: add links to good "best practices" reading/watching material, as well as my own suggestions
- TODO: add checklist on how to approach a new problem
- TODO: add extra suggested libraries (nbautoexport, ...)
############################################################################ -->
