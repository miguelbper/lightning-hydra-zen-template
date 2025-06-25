from hydra_zen import launch, store, zen
from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.configs import EvalCfg, TrainCfg

store.add_to_hydra_store()


def mock_train(
    data: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: str | None = None,
    evaluate: bool = True,
    matmul_precision: str | None = None,
    compile: bool = False,
) -> None:
    pass


def mock_eval(data: LightningDataModule, model: LightningModule, trainer: Trainer, ckpt_path: str) -> None:
    pass


def test_train_config() -> None:
    launch(TrainCfg, zen(mock_train), version_base="1.3")


def test_evaluate_config() -> None:
    launch(EvalCfg, zen(mock_eval), version_base="1.3", ckpt_path="")
