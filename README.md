<div align="center">

# Lightning hydra-zen Template
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![PyTorch Lightning](https://img.shields.io/badge/-Lightning-7e4fff?logo=lightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Configs-Hydra-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) <br>
[![Code Quality](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/lightning-hydra-zen-template/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/lightning-hydra-zen-template/branch/main/graph/badge.svg)](https://codecov.io/gh/miguelbper/lightning-hydra-zen-template)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/miguelbper/lightning-hydra-zen-template/blob/main/LICENSE)

A template for Deep Learning projects using PyTorch Lightning and hydra-zen

Click on [<kbd>Use this template</kbd>](https://github.com/miguelbper/lightning-hydra-zen-template/generate) to start a new project!

![img.png](img.png)

</div>

---
<!-- TODO: add better description -->
## Description

A template for Deep Learning projects, using modern tooling and practices. Implements the features which are common across different Deep Learning problems (logging, checkpointing, experiment tracking, training and evaluation scripts...) so that you can focus on the specifics of your problem: data analysis and modeling.

Goals of this template:
- Have a low amount of code relative to the functionality being offered
- Be easy to extend by adding new datasets or models, without requiring modifications to the existing code
- Offer a logical and simple to understand folder structure
- Promote best practices for Machine Learning experiment **correctness** and **reproducibility**

Tech Stack:
- [PyTorch](https://github.com/pytorch/pytorch) - Tensors and neural networks
- [Lightning](https://github.com/Lightning-AI/pytorch-lightning) - Framework for organizing PyTorch code
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) - Metrics for Lightning
- [Optuna](https://github.com/optuna/optuna) - Hyperparameter optimization
- [MLflow](https://github.com/mlflow/mlflow) - Experiment tracking
- [Hydra](https://github.com/facebookresearch/hydra) - Configuration files
- [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) - Wrapper for Hydra
- [Ruff](https://github.com/astral-sh/ruff) - Linting and formatting
- [uv](https://github.com/astral-sh/uv) - Dependency management

## Directory structure
```
├── .github/                           <- GitHub Actions workflows
│   └── workflows/
│       ├── code-quality.yaml
│       ├── coverage.yaml
│       ├── publish.yaml
│       └── tests.yaml
│
├── data/                              <- Directory for datasets
│   ├── external/                      <- External data sources
│   ├── interim/                       <- Intermediate results of dataset processing
│   ├── processed/                     <- Datasets ready to be used by the modelling scripts
│   └── raw/                           <- Datasets as obtained from the source
│
├── logs/                              <- Training logs, artifacts, metrics, checkpoints, and experiment tracking data
│
├── notebooks/                         <- Jupyter notebooks for experimentation
│
├── scripts/                           <- Shell scripts
│
├── src/                               <- Source code for the project
│   └── lightning_hydra_zen_template/  <- Main package directory
│       ├── configs/                   <- Configuration files for Hydra
│       │   ├── groups/
│       │   │   ├── __init__.py
│       │   │   ├── callbacks.py       <- Callback configurations
│       │   │   ├── data.py            <- Data module configurations
│       │   │   ├── debug.py           <- Debug configurations
│       │   │   ├── experiment.py      <- Experiment configurations
│       │   │   ├── hparams_search.py  <- Hyperparameter search configurations
│       │   │   ├── hydra_conf.py      <- Hydra configuration settings
│       │   │   ├── logger.py          <- Logger configurations
│       │   │   ├── model.py           <- Model configurations
│       │   │   ├── paths.py           <- Path configurations
│       │   │   └── trainer.py         <- Trainer configurations
│       │   │
│       │   ├── utils/                 <- Utility functions for configurations
│       │   ├── __init__.py
│       │   ├── eval.py                <- Main configuration for evaluation
│       │   └── train.py               <- Main configuration for training
│       │
│       ├── data/                      <- LightningDataModules for handling datasets
│       │
│       ├── model/                     <- LightningModules
│       │
│       ├── utils/                     <- Utility functions
│       │
│       ├── __init__.py
│       ├── eval.py                    <- Main testing / evaluation script
│       └── train.py                   <- Main training script
│
├── tests/                             <- Automated tests
│
├── .envrc.example                     <- Example environment variables file (rename to .envrc)
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version                    <- Python version that will be installed
├── img.png
├── justfile                           <- Project commands
├── LICENSE
├── pyproject.toml                     <- Project configuration file with dependencies and tool settings
├── README.md
└── uv.lock                            <- The requirements file for reproducing the environment
```

## Reproducibility of ML experiments

One of the goals of this template is to help users in creating **correct** and **reproducible** ML code.

Libraries like Hydra and hydra-zen help with this objective:
- We can create configuration files which are hierarchical and composable/overridable.
- With hydra-zen we can automatically create configurations for the full signature of an object without code duplication.
- This results in a resolved configuration with full information about the experiment that was run.
- The hierarchy of the configuration completely mirrors that of the classes being used, which makes it easy to understand.

Here is an example of a resolved config file (that is printed to the terminal and the `logs/` directory).

<details>
<summary>Click to expand the resolved config</summary>

```
config
├── data
│   ├── _target_: hydra_zen.funcs.zen_processing
│   ├── _zen_target: lightning_hydra_zen_template.data.mnist.MNISTDataModule
│   ├── _zen_wrappers: lightning_hydra_zen_template.configs.utils.utils.log_instantiation
│   ├── data_dir: /path/to/lightning-hydra-zen-template/data/raw
│   ├── batch_size: 32
│   ├── num_workers: 0
│   ├── pin_memory: False
│   └── num_val_examples: 5000
├── model
│   ├── _target_: hydra_zen.funcs.zen_processing
│   ├── _zen_target: lightning_hydra_zen_template.model.model.Model
│   ├── _zen_wrappers: lightning_hydra_zen_template.configs.utils.utils.log_instantiation
│   ├── net
│   │   ├── _target_: lightning_hydra_zen_template.model.components.resnet.ResNet
│   │   └── num_classes: 10
│   ├── loss_fn
│   │   ├── _target_: torch.nn.modules.loss.CrossEntropyLoss
│   │   ├── weight: None
│   │   ├── size_average: None
│   │   ├── ignore_index: -100
│   │   ├── reduce: None
│   │   ├── reduction: mean
│   │   └── label_smoothing: 0.0
│   ├── optimizer
│   │   ├── _target_: torch.optim.adam.Adam
│   │   ├── _partial_: True
│   │   ├── lr: 0.001
│   │   ├── betas
│   │   │   ├── 0: 0.9
│   │   │   └── 1: 0.999
│   │   ├── eps: 1e-08
│   │   ├── weight_decay: 0.0
│   │   ├── amsgrad: False
│   │   ├── foreach: None
│   │   ├── maximize: False
│   │   ├── capturable: False
│   │   ├── differentiable: False
│   │   ├── fused: None
│   │   └── decoupled_weight_decay: False
│   ├── scheduler
│   │   ├── _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
│   │   ├── _partial_: True
│   │   ├── mode: min
│   │   ├── factor: 0.1
│   │   ├── patience: 10
│   │   ├── threshold: 0.0001
│   │   ├── threshold_mode: rel
│   │   ├── cooldown: 0
│   │   ├── min_lr: 0
│   │   └── eps: 1e-08
│   └── metric_collection
│       ├── _target_: torchmetrics.collections.MetricCollection
│       ├── metrics
│       │   ├── 0
│       │   │   ├── _target_: torchmetrics.classification.accuracy.Accuracy
│       │   │   ├── task: multiclass
│       │   │   ├── threshold: 0.5
│       │   │   ├── num_classes: 10
│       │   │   ├── num_labels: None
│       │   │   ├── average: micro
│       │   │   ├── multidim_average: global
│       │   │   ├── top_k: 1
│       │   │   ├── ignore_index: None
│       │   │   └── validate_args: True
│       │   ├── 1
│       │   │   ├── _target_: torchmetrics.classification.f_beta.F1Score
│       │   │   ├── task: multiclass
│       │   │   ├── threshold: 0.5
│       │   │   ├── num_classes: 10
│       │   │   ├── num_labels: None
│       │   │   ├── average: macro
│       │   │   ├── multidim_average: global
│       │   │   ├── top_k: 1
│       │   │   ├── ignore_index: None
│       │   │   ├── validate_args: True
│       │   │   └── zero_division: 0.0
│       │   ├── 2
│       │   │   ├── _target_: torchmetrics.classification.precision_recall.Precision
│       │   │   ├── task: multiclass
│       │   │   ├── threshold: 0.5
│       │   │   ├── num_classes: 10
│       │   │   ├── num_labels: None
│       │   │   ├── average: macro
│       │   │   ├── multidim_average: global
│       │   │   ├── top_k: 1
│       │   │   ├── ignore_index: None
│       │   │   └── validate_args: True
│       │   └── 3
│       │       ├── _target_: torchmetrics.classification.precision_recall.Recall
│       │       ├── task: multiclass
│       │       ├── threshold: 0.5
│       │       ├── num_classes: 10
│       │       ├── num_labels: None
│       │       ├── average: macro
│       │       ├── multidim_average: global
│       │       ├── top_k: 1
│       │       ├── ignore_index: None
│       │       └── validate_args: True
│       ├── prefix: None
│       ├── postfix: None
│       └── compute_groups: True
├── trainer
│   ├── _target_: hydra_zen.funcs.zen_processing
│   ├── _zen_target: lightning.pytorch.trainer.trainer.Trainer
│   ├── _zen_wrappers: lightning_hydra_zen_template.configs.utils.utils.log_instantiation
│   ├── accelerator: auto
│   ├── strategy: auto
│   ├── devices: auto
│   ├── num_nodes: 1
│   ├── precision: None
│   ├── logger
│   │   ├── 0
│   │   │   ├── _target_: lightning.pytorch.loggers.csv_logs.CSVLogger
│   │   │   ├── save_dir: /path/to/lightning-hydra-zen-template/logs/train/runs/2025-05-09/22-32-03
│   │   │   ├── name: csv
│   │   │   ├── version: None
│   │   │   ├── prefix:
│   │   │   └── flush_logs_every_n_steps: 100
│   │   ├── 1
│   │   │   ├── _target_: lightning.pytorch.loggers.tensorboard.TensorBoardLogger
│   │   │   ├── save_dir: /path/to/lightning-hydra-zen-template/logs/train/runs/2025-05-09/22-32-03
│   │   │   ├── name: tensorboard
│   │   │   ├── version: None
│   │   │   ├── log_graph: False
│   │   │   ├── default_hp_metric: True
│   │   │   ├── prefix:
│   │   │   └── sub_dir: None
│   │   └── 2
│   │       ├── _target_: lightning.pytorch.loggers.mlflow.MLFlowLogger
│   │       ├── experiment_name: lightning_logs
│   │       ├── run_name: None
│   │       ├── tracking_uri: /path/to/lightning-hydra-zen-template/logs/mlflow/mlruns
│   │       ├── tags: None
│   │       ├── save_dir: ./mlruns
│   │       ├── log_model: False
│   │       ├── checkpoint_path_prefix:
│   │       ├── prefix:
│   │       ├── artifact_location: None
│   │       ├── run_id: None
│   │       └── synchronous: None
│   ├── callbacks
│   │   ├── 0
│   │   │   ├── _target_: lightning.pytorch.callbacks.progress.rich_progress.RichProgressBar
│   │   │   ├── refresh_rate: 1
│   │   │   ├── leave: False
│   │   │   ├── theme
│   │   │   │   ├── _target_: lightning.pytorch.callbacks.progress.rich_progress.RichProgressBarTheme
│   │   │   │   ├── description:
│   │   │   │   ├── progress_bar: #6206E0
│   │   │   │   ├── progress_bar_finished: #6206E0
│   │   │   │   ├── progress_bar_pulse: #6206E0
│   │   │   │   ├── batch_progress:
│   │   │   │   ├── time: dim
│   │   │   │   ├── processing_speed: dim underline
│   │   │   │   ├── metrics: italic
│   │   │   │   ├── metrics_text_delimiter:
│   │   │   │   └── metrics_format: .3f
│   │   │   └── console_kwargs: None
│   │   ├── 1
│   │   │   ├── _target_: lightning.pytorch.callbacks.rich_model_summary.RichModelSummary
│   │   │   └── max_depth: 1
│   │   ├── 2
│   │   │   ├── _target_: lightning.pytorch.callbacks.early_stopping.EarlyStopping
│   │   │   ├── monitor: val/MulticlassAccuracy
│   │   │   ├── min_delta: 0.0
│   │   │   ├── patience: 3
│   │   │   ├── verbose: False
│   │   │   ├── mode: max
│   │   │   ├── strict: True
│   │   │   ├── check_finite: True
│   │   │   ├── stopping_threshold: None
│   │   │   ├── divergence_threshold: None
│   │   │   ├── check_on_train_epoch_end: None
│   │   │   └── log_rank_zero_only: False
│   │   ├── 3
│   │   │   ├── _target_: lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint
│   │   │   ├── dirpath: /path/to/lightning-hydra-zen-template/logs/train/runs/2025-05-09/22-32-03/checkpoints
│   │   │   ├── filename: epoch_{epoch:03d}
│   │   │   ├── monitor: val/MulticlassAccuracy
│   │   │   ├── verbose: False
│   │   │   ├── save_last: True
│   │   │   ├── save_top_k: 1
│   │   │   ├── save_weights_only: False
│   │   │   ├── mode: max
│   │   │   ├── auto_insert_metric_name: False
│   │   │   ├── every_n_train_steps: None
│   │   │   ├── train_time_interval: None
│   │   │   ├── every_n_epochs: None
│   │   │   ├── save_on_train_epoch_end: None
│   │   │   └── enable_version_counter: True
│   │   └── 4
│   │       └── _target_: lightning_hydra_zen_template.utils.print_config.LogConfigToMLflow
│   ├── fast_dev_run: False
│   ├── max_epochs: 1
│   ├── min_epochs: 1
│   ├── max_steps: -1
│   ├── min_steps: None
│   ├── max_time: None
│   ├── limit_train_batches: None
│   ├── limit_val_batches: None
│   ├── limit_test_batches: None
│   ├── limit_predict_batches: None
│   ├── overfit_batches: 0.0
│   ├── val_check_interval: None
│   ├── check_val_every_n_epoch: 1
│   ├── num_sanity_val_steps: None
│   ├── log_every_n_steps: None
│   ├── enable_checkpointing: None
│   ├── enable_progress_bar: None
│   ├── enable_model_summary: False
│   ├── accumulate_grad_batches: 1
│   ├── gradient_clip_val: None
│   ├── gradient_clip_algorithm: None
│   ├── deterministic: False
│   ├── benchmark: None
│   ├── inference_mode: True
│   ├── use_distributed_sampler: True
│   ├── profiler: None
│   ├── detect_anomaly: False
│   ├── barebones: False
│   ├── plugins: None
│   ├── sync_batchnorm: False
│   ├── reload_dataloaders_every_n_epochs: 0
│   ├── default_root_dir: /path/to/lightning-hydra-zen-template/logs/train/runs/2025-05-09/22-32-03
│   └── model_registry: None
├── task_name: train
├── tags
│   └── 0: dev
├── evaluate: True
├── ckpt_path: None
├── seed: 42
├── monitor: val/MulticlassAccuracy
├── mode: max
├── matmul_precision: None
└── compile: None
```
</details>


## Installation

```bash
# Install dependencies
uv sync

# Run unit tests
uv run pytest
```

## Usage

To train or evaluate a model on MNIST, do:
```bash
# Train
uv run src/lightning_hydra_zen_template/train.py

# Evaluate
uv run src/lightning_hydra_zen_template/eval.py ckpt_path=...
```

To define a new experiment, it is necessary to:
- Define the dataset, by creating a `LightningDataModule` in `data/`
- Define the model, by creating a `LightningModule` in `model/`
- Define a configuration for this experiment, by adding new config group options to `configs/groups/data.py` and `configs/groups/model.py`
- Or, create a new experiment config in `configs/groups/experiment.py` by adding the group options defined above to this new experiment

With this done, you can train with the data and model specified above by overriding the corresponding groups from the command line:
```bash
# Override data and model groups individually
uv run src/lightning_hydra_zen_template/train.py data=new_datamodule model=new_model

# Or, override with a new experiment config
uv run src/lightning_hydra_zen_template/train.py experiment=new_experiment
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This template is heavily inspired by [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) and it is my personal take on that idea. Another very nice template for data science projects is [drivendataorg/cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science).
