from hydra_zen import make_config, store

from lightning_hydra_zen_template.configs.utils.paths import output_dir
from lightning_hydra_zen_template.configs.utils.utils import remove_types

SKLearnExperimentCfg = make_config(
    hydra_defaults=[
        {"override /model": "logistic"},
        {"override /trainer": "sklearn"},
        {"override /callbacks": None},
        {"override /logger": None},
        "_self_",
    ],
    task_name="sklearn",
    monitor="val/accuracy_score",
    mode="max",
    output_dir=output_dir,
    experiment_name="mnist",
    run_name="logistic",
)


experiment_store = store(group="experiment", package="_global_", to_config=remove_types)
experiment_store(SKLearnExperimentCfg, name="sklearn")
