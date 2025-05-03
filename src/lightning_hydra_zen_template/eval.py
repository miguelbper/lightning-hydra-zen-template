import logging

from hydra_zen import store, zen
from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.configs import EvalCfg
from lightning_hydra_zen_template.utils.print_config import print_config

log = logging.getLogger(__name__)


def evaluate(
    data: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: str,
) -> None:
    """Test a model from a configuration object (which should include a
    checkpoint).

    Args:
        data (LightningDataModule): The data module to use for evaluation
        model (LightningModule): The model to evaluate
        trainer (Trainer): The trainer to use for evaluation
        ckpt_path (str): Path to the checkpoint to load
    """
    log.info("Testing model")
    trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)


def main() -> None:
    store(EvalCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(evaluate, pre_call=print_config)
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
