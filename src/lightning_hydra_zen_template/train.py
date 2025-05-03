import logging

import lightning as L
from hydra_zen import store, zen

from lightning_hydra_zen_template.configs import TrainCfg
from lightning_hydra_zen_template.utils.print_config import print_config

log = logging.getLogger(__name__)


def train() -> float:
    log.info("Training model")
    return 1


def seed_fn(seed: int) -> None:
    log.info(f"Setting seed to {seed}")
    L.seed_everything(seed, workers=True, verbose=False)


def main() -> None:
    store(TrainCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=[zen(seed_fn), print_config])
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
