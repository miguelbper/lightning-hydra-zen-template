import logging

import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT

log = logging.getLogger(__name__)


def train(
    data: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: str | None = None,
    matmul_precision: str | None = None,
    compile: bool = False,
    return_all_metrics: bool = False,
) -> float | dict[str, float]:
    """Train, validate and test a PyTorch Lightning model.

    Args:
        data (LightningDataModule): The data module containing training, validation and test data.
        model (LightningModule): The PyTorch Lightning model to train.
        trainer (Trainer): The PyTorch Lightning trainer instance.
        ckpt_path (str | None, optional): Path to a checkpoint to resume training from. Defaults to None.
        matmul_precision (str | None, optional): Precision for matrix multiplication. Defaults to None.
        compile (bool | None, optional): Whether to compile the model using torch.compile(). Defaults to True.
        return_all_metrics (bool, optional): Whether to return all metrics or just the best one. Defaults to False.

    Returns:
        float: The best model score achieved during training, or None if no score was recorded.
    """
    if not hasattr(trainer, "checkpoint_callback"):
        raise ValueError("Trainer does not have a checkpoint callback")
    if not compatible_inputs(data=data, model=model, trainer=trainer):
        raise ValueError("Data, model and trainer must be compatible")

    if matmul_precision:
        log.info(f"Setting matmul precision to {matmul_precision}")
        torch.set_float32_matmul_precision(matmul_precision)

    if compile and isinstance(model, torch.nn.Module):
        log.info("Compiling model")
        model = torch.compile(model)

    log.info("Training model")
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)
    ckpt_path: str = trainer.checkpoint_callback.best_model_path
    metric: float = trainer.checkpoint_callback.best_model_score.item()

    log.info("Validating model")
    val_out: _EVALUATE_OUTPUT = trainer.validate(model=model, datamodule=data, ckpt_path=ckpt_path)

    log.info("Testing model")
    test_out: _EVALUATE_OUTPUT = trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)

    metrics: dict[str, float] = {**val_out[0], **test_out[0]}
    return metrics if return_all_metrics else metric


def seed_fn(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def compatible_inputs(data: LightningDataModule, model: LightningModule, trainer: Trainer) -> bool:
    """Check if the data, model and trainer are compatible.

    Args:
        data (LightningDataModule): The data module containing training, validation and test data.
        model (LightningModule): The PyTorch Lightning model to train.
        trainer (Trainer): The PyTorch Lightning trainer instance.
    """
    return True
