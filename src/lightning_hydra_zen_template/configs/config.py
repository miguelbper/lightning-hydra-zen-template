import os

from hydra.conf import HydraConf, RunDir, SweepDir
from hydra_zen import ZenField as zf
from hydra_zen import make_config, store
from rootutils import find_root

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

root_dir = str(find_root(search_from=__file__))
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")
output_dir = "${hydra:runtime.output_dir}"
work_dir = "${hydra:runtime.cwd}"

year_month_day = "${now:%Y-%m-%d}"
hour_minute_second = "${now:%H-%M-%S}"
task_name = "${task_name}"

run_dir = os.path.join(log_dir, task_name, "runs", year_month_day, hour_minute_second)
sweep_dir = os.path.join(log_dir, task_name, "multiruns", year_month_day, hour_minute_second)
job_file = os.path.join(output_dir, f"{task_name}.log")

PathsCfg = make_config(
    root_dir=zf(str, root_dir),
    data_dir=zf(str, data_dir),
    log_dir=zf(str, log_dir),
    output_dir=zf(str, output_dir),
    work_dir=zf(str, work_dir),
)

# ------------------------------------------------------------------------------
# Hydra
# ------------------------------------------------------------------------------

HydraCfg = HydraConf(
    run=RunDir(run_dir),
    sweep=SweepDir(dir=sweep_dir, subdir="${hydra:job.num}"),
    # Fix from PR https://github.com/facebookresearch/hydra/pull/2242, while there isn't a new release
    job_logging={"handlers": {"file": {"filename": job_file}}},
)

store(HydraCfg)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

TrainCfg = make_config(
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
