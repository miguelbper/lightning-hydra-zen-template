import logging

import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer

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
