import logging

import lightning as L
from hydra_zen import store, zen

from lightning_hydra_zen_template.configs import Config

log = logging.getLogger(__name__)


def train() -> None:
    log.info("Training model")


def seed_fn(seed: int) -> None:
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def main() -> None:
    store(Config, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=zen(seed_fn))
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
