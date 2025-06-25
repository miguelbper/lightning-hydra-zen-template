from hydra_zen import launch, store, zen
from hydra_zen._launch import OverrideDict

from lightning_hydra_zen_template.configs import TrainCfg
from lightning_hydra_zen_template.funcs.train import train


class TestTrain:
    def test_train(self, matrix_overrides: OverrideDict) -> None:
        store.add_to_hydra_store()
        launch(TrainCfg, zen(train), version_base="1.3", overrides=matrix_overrides)

    def test_train_with_callbacks(self, overrides: OverrideDict) -> None:
        store.add_to_hydra_store()
        overrides = overrides.copy()
        overrides.pop("trainer.callbacks")
        overrides.pop("trainer.logger")
        launch(TrainCfg, zen(train), version_base="1.3", overrides=overrides)
