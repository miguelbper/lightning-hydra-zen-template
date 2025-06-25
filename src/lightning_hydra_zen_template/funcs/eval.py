import logging

from lightning import LightningDataModule, LightningModule, Trainer

log = logging.getLogger(__name__)


def evaluate(data: LightningDataModule, model: LightningModule, trainer: Trainer, ckpt_path: str) -> None:
    """Evaluate a trained model using a checkpoint.

    This function loads a model from a checkpoint and runs evaluation on the test set
    using the provided data module and trainer.

    Args:
        data (LightningDataModule): The data module containing test data.
        model (LightningModule): The PyTorch Lightning model to evaluate.
        trainer (Trainer): The PyTorch Lightning trainer instance.
        ckpt_path (str): Path to the checkpoint file to load the model from.
    """
    log.info("Testing model")
    trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
