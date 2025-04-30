from hydra_zen import make_config

from lightning_hydra_zen_template.configs.paths.default import PathsCfg

Config = make_config(
    hydra_defaults=[
        "_self_",
        {"override hydra/hydra_logging": "colorlog"},
        {"override hydra/job_logging": "colorlog"},
    ],
    paths=PathsCfg,
    task_name="train",
    tags=["dev"],
    evaluate=True,
    ckpt_path=None,
    seed=42,
)
