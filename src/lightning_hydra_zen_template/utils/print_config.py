import copy
import logging
import os
from itertools import chain

import rich
import rich.syntax
import rich.tree
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.client import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf, flag_override

log = logging.getLogger(__name__)


def print_config(cfg: DictConfig) -> None:
    """Print and save the configuration tree to a file.

    This function takes a Hydra configuration object, removes specified packages,
    converts it to a rich tree structure, and saves it to a config_tree.log file
    in the output directory.

    Args:
        cfg (DictConfig): The Hydra configuration object to print and save.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    log.info(f"Output directory is {cfg.paths.output_dir}")
    config_file = os.path.join(cfg.paths.output_dir, "config_tree.log")

    cfg = remove_packages(cfg)
    tree = cfg_to_tree(cfg)

    log.info(f"Saving config to {config_file}")
    with open(config_file, "w") as file:
        rich.print(tree, file=file)
    rich.print(tree)


def remove_packages(cfg: DictConfig, packages: tuple[str, ...] = ("hydra", "paths")) -> DictConfig:
    """Remove specified packages from the configuration object.

    Args:
        cfg (DictConfig): The configuration object to modify.
        packages (tuple[str, ...], optional): Tuple of package names to remove.
            Defaults to ("hydra", "paths").

    Returns:
        DictConfig: A copy of the configuration object with specified packages removed.
    """
    for package in packages:
        if package in cfg:
            cfg = copy.copy(cfg)
            with flag_override(cfg, ["struct", "readonly"], [False, False]):
                cfg.pop(package)
    return cfg


def cfg_to_tree(cfg: DictConfig) -> rich.tree.Tree:
    """Convert a configuration object to a rich tree structure.

    Args:
        cfg (DictConfig): The configuration object to convert.

    Returns:
        rich.tree.Tree: A rich tree structure representing the configuration.
    """

    def add_to_tree(tree: rich.tree.Tree, level: int, cfg: DictConfig) -> None:
        colors = ["red", "cyan", "blue", "green"]
        color = colors[min(level, len(colors) - 1)]
        for key, value in cfg.items():
            key_format = f"[bold {color}]{key}[/bold {color}]"
            if isinstance(value, DictConfig):
                sub_tree = tree.add(key_format)
                add_to_tree(sub_tree, level + 1, value)
            elif isinstance(value, ListConfig):
                sub_tree = tree.add(key_format)
                add_to_tree(sub_tree, level + 1, OmegaConf.create({str(i): item for i, item in enumerate(value)}))
            else:
                tree.add(f"{key_format}: {value}")

    tree = rich.tree.Tree("config")
    add_to_tree(tree, 0, cfg)
    return tree


class LogConfigToMLflow(Callback):
    """A PyTorch Lightning callback that logs configuration files to MLflow.

    This callback logs all configuration files from the output directory and
    .hydra directory to MLflow when training starts. It requires an MLflow
    logger to be present in the trainer's loggers.

    Attributes:
        None
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log configuration files to MLflow when training starts.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The PyTorch Lightning module being trained.
        """
        mlf_logger: MLFlowLogger | None = next((lg for lg in trainer.loggers if isinstance(lg, MLFlowLogger)), None)
        if mlf_logger is None:
            log.warning("No MLflow logger found, skipping logging to MLflow")
            return
        mlf_client: MlflowClient = mlf_logger.experiment
        run_id: str = mlf_logger.run_id
        output_dir: str = trainer.default_root_dir
        hydra_dir: str = os.path.join(output_dir, ".hydra")
        dirs: list[str] = [output_dir, hydra_dir]
        entries: list[os.DirEntry] = list(chain.from_iterable(os.scandir(dir) for dir in dirs))
        files: list[str] = [entry.path for entry in entries if entry.is_file()]
        log.info("Logging files to MLflow")
        for file_path in files:
            mlf_client.log_artifact(run_id, file_path)
