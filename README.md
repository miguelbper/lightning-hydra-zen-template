# Deep Learning Template
[![python](https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<!-- add automatic tests passing -->
<!-- add automatic code coverage -->
[![license](https://img.shields.io/badge/license-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)  <!-- automate -->



Template for deep learning projects, using
- [PyTorch](https://github.com/pytorch/pytorch) - Neural networks
- [Lightning](https://github.com/Lightning-AI/pytorch-lightning) - PyTorch training loop abstraction
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) - Metrics for Lightning
- [TensorBoard](https://github.com/tensorflow/tensorboard) - Training monitoring
- [Optuna](https://github.com/optuna/optuna) - Hyperparameter optimization
- [MLflow](https://github.com/mlflow/mlflow) - Experiment tracking
- [Hydra](https://github.com/facebookresearch/hydra) - Configuration files
- [Loguru](https://github.com/Delgan/loguru) - Logging

## Setup
Fork the repo or copy the code to the working directory of the new project.

Create a conda environment:
```bash
conda env create -f environment.yaml
```
