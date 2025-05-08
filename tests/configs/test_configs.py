from hydra_zen import launch, store, zen
from lightning import LightningDataModule, LightningModule, Trainer

from lightning_hydra_zen_template.configs import EvalCfg, TrainCfg

store.add_to_hydra_store()


def mock(
    data: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    ckpt_path: str | None = None,
    evaluate: bool | None = True,
    matmul_precision: str | None = None,
    compile: bool | None = True,
) -> float:
    pass


def test_train_config() -> None:
    launch(TrainCfg, zen(mock), version_base="1.3")


def test_evaluate_config() -> None:
    launch(EvalCfg, zen(mock), version_base="1.3", ckpt_path="")
