import logging

import lightning as L
import torch
from hydra_zen import store, zen
from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.configs import TrainCfg
from lightning_hydra_zen_template.utils.logging import log_git_status, log_python_env, print_config

log = logging.getLogger(__name__)


def train(
    data: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: str | None = None,
    evaluate: bool | None = True,
    matmul_precision: str | None = None,
    compile: bool | None = True,
) -> float:
    """Train, validate and test a PyTorch Lightning model.

    Args:
        data (LightningDataModule): The data module containing training, validation and test data.
        model (LightningModule): The PyTorch Lightning model to train.
        trainer (Trainer): The PyTorch Lightning trainer instance.
        ckpt_path (str | None, optional): Path to a checkpoint to resume training from. Defaults to None.
        evaluate (bool | None, optional): Whether to run validation and testing after training. Defaults to True.
        matmul_precision (str | None, optional): Precision for matrix multiplication. Defaults to None.
        compile (bool | None, optional): Whether to compile the model using torch.compile(). Defaults to True.

    Returns:
        float: The best model score achieved during training, or None if no score was recorded.
    """
    if matmul_precision:
        log.info(f"Setting matmul precision to {matmul_precision}")
        torch.set_float32_matmul_precision(matmul_precision)

    if compile:
        log.info("Compiling model")
        model = torch.compile(model)

    log.info("Training model")
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)
    metric: torch.Tensor | None = trainer.checkpoint_callback.best_model_score
    ckpt_path: str = trainer.checkpoint_callback.best_model_path

    if evaluate and ckpt_path:
        log.info("Validating model")
        trainer.validate(model=model, datamodule=data, ckpt_path=ckpt_path)

        log.info("Testing model")
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)

    return metric.item() if metric is not None else None


def seed_fn(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def main() -> None:
    """Main entry point for the training script.

    Sets up Hydra configuration and runs the training task with the
    specified configuration.
    """
    store(TrainCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=[log_git_status, log_python_env, zen(seed_fn), print_config])
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
