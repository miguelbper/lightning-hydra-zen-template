from hydra_plugins.hydra_optuna_sweeper.config import OptunaSweeperConf, TPESamplerConfig
from hydra_zen import make_config, store

from lightning_hydra_zen_template.configs.utils.utils import remove_types

# https://github.com/mit-ll-responsible-ai/hydra-zen/issues/563

TPESamplerCfg = make_config(
    bases=(TPESamplerConfig,),
    seed=1234,
    n_startup_trials=10,
)
store(TPESamplerCfg, group="hydra/sweeper/sampler", name="custom_tpe")

OptunaSweeperCfg = make_config(
    hydra_defaults=["_self_", {"sampler": "custom_tpe"}],
    bases=(OptunaSweeperConf,),
    storage=None,
    study_name=None,
    n_jobs=1,
    direction="${mode}imize",
    n_trials=10,
    params={
        "model.optimizer.lr": "interval(0.0001, 0.1)",
        "data.batch_size": "choice(32, 64, 128, 256)",
    },
)
store(OptunaSweeperCfg, group="hydra/sweeper", name="custom_optuna")

HparamsSearchOptunaCfg = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "custom_optuna"}],
    hydra=dict(mode="MULTIRUN"),
)
hparams_store = store(group="hparams_search", package="_global_", to_config=remove_types)
hparams_store(HparamsSearchOptunaCfg, name="mnist")
