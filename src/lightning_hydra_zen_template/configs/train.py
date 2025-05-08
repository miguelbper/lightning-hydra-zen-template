from hydra_zen import make_config

from lightning_hydra_zen_template.configs.groups.paths import PathsCfg

TrainCfg = make_config(
    hydra_defaults=[
        "_self_",
        # Main components
        {"data": "mnist"},
        {"model": "mnist"},
        {"trainer": "default"},
        {"logger": "default"},
        {"callbacks": "default"},
        # Overrides to main components
        {"experiment": None},
        {"hparams_search": None},
        {"debug": None},
        # Colored logging
        {"override hydra/hydra_logging": "colorlog"},
        {"override hydra/job_logging": "colorlog"},
    ],
    # Main components
    data=None,
    model=None,
    trainer=None,
    # Run configs
    paths=PathsCfg,
    task_name="train",
    tags=["dev"],
    evaluate=True,
    ckpt_path=None,
    seed=42,
    monitor="val/MulticlassAccuracy",
    mode="max",
    matmul_precision=None,
    compile=None,
)
