from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper
from hydra_zen import make_config, store
from optuna.samplers import TPESampler

from lightning_hydra_zen_template.configs.utils.utils import fbuilds, remove_types

HparamsSearchOptunaCfg = make_config(
    hydra=dict(
        mode="MULTIRUN",
        sweeper=fbuilds(
            OptunaSweeper,
            storage=None,
            study_name=None,
            n_jobs=1,
            direction="${mode}imize",
            n_trials=20,
            sampler=fbuilds(
                TPESampler,
                seed=1234,
                n_startup_trials=10,
            ),
            params={
                "model.optimizer.lr": "interval(0.0001, 0.1)",
                "data.batch_size": "choice(32, 64, 128, 256)",
            },
        ),
    ),
)

hparams_store = store(group="hparams_search", package="_global_", to_config=remove_types)
hparams_store(HparamsSearchOptunaCfg, name="mnist_optuna")
