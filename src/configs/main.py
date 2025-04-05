import lightning as L
from hydra_zen import ZenStore, to_yaml, zen
from lightning import LightningDataModule, LightningModule, Trainer

from src.configs.config import Config


def train_mock(datamodule: LightningDataModule, model: LightningModule, trainer: Trainer) -> None:
    print(datamodule)
    print(model)
    print(trainer)
    # trainer.fit(model=model, datamodule=datamodule)


def main() -> None:
    print(to_yaml(Config))
    store = ZenStore(deferred_hydra_store=False)
    store(Config, name="my_config")
    pre_seed = zen(lambda seed: L.seed_everything(seed, workers=True))
    task_fn = zen(train_mock, pre_call=pre_seed)
    task_fn.hydra_main(config_name="my_config", version_base="1.3", config_path=".")


if __name__ == "__main__":
    main()
