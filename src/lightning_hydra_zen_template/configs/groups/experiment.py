from hydra_zen import make_config, store

from lightning_hydra_zen_template.configs.utils.utils import remove_types

ExperimentExampleCfg = make_config(
    hydra_defaults=[
        {"override /data": "mnist"},
        {"override /model": "mnist"},
        "_self_",
    ],
    data=dict(
        batch_size=64,
    ),
    model=dict(
        optimizer=dict(
            lr=0.002,
        ),
    ),
    trainer=dict(
        min_epochs=10,
        max_epochs=10,
        gradient_clip_val=0.5,
    ),
    seed=0,
)


experiment_store = store(group="experiment", package="_global_", to_config=remove_types)
experiment_store(ExperimentExampleCfg, name="example")
