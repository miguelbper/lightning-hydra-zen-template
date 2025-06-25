from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.scikit_learn.datamodule import SKLearnDataModule
from lightning_hydra_zen_template.scikit_learn.module import SKLearnModule
from lightning_hydra_zen_template.scikit_learn.trainer import SKLearnTrainer


def valid_types(
    data: LightningDataModule | SKLearnDataModule,
    model: LightningModule | SKLearnModule,
    trainer: Trainer | SKLearnTrainer,
) -> bool:
    """Validate that trainer, model, and datamodule types are consistent.

    This function ensures that if the trainer is Lightning, then both model and datamodule
    are also Lightning components. Similarly, if the trainer is SKLearn, then both model
    and datamodule should be SKLearn components.

    Args:
        data (LightningDataModule | SKLearnDataModule): The data module to validate.
        model (LightningModule | SKLearnModule): The model to validate.
        trainer (Trainer | SKLearnTrainer): The trainer to validate.

    Returns:
        bool: True if all components are of the same type (all Lightning or all SKLearn),
              False otherwise.
    """
    if isinstance(trainer, Trainer):
        return isinstance(data, LightningDataModule) and isinstance(model, LightningModule)
    elif isinstance(trainer, SKLearnTrainer):
        return isinstance(data, SKLearnDataModule) and isinstance(model, SKLearnModule)
    return False
