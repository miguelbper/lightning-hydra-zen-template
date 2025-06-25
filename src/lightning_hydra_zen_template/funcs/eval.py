import logging

from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.funcs.validate import valid_types
from lightning_hydra_zen_template.scikit_learn.datamodule import SKLearnDataModule
from lightning_hydra_zen_template.scikit_learn.module import SKLearnModule
from lightning_hydra_zen_template.scikit_learn.trainer import SKLearnTrainer

log = logging.getLogger(__name__)


def evaluate(
    data: LightningDataModule | SKLearnDataModule,
    model: LightningModule | SKLearnModule,
    trainer: Trainer | SKLearnTrainer,
    ckpt_path: str,
) -> None:
    """Evaluate a trained model using a checkpoint.

    This function loads a model from a checkpoint and runs evaluation on the test set
    using the provided data module and trainer. Supports both PyTorch Lightning and
    scikit-learn components.

    Args:
        data (LightningDataModule | SKLearnDataModule): The data module containing test data.
        model (LightningModule | SKLearnModule): The model to evaluate.
        trainer (Trainer | SKLearnTrainer): The trainer instance.
        ckpt_path (str): Path to the checkpoint file to load the model from.

    Returns:
        None: This function performs evaluation but does not return any values.
    """
    if not valid_types(data=data, model=model, trainer=trainer):
        raise ValueError("Invalid types for data, model, and trainer")

    log.info("Testing model")
    trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
