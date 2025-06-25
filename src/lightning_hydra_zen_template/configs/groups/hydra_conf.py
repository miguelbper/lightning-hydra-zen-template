import os

from hydra.conf import HydraConf, RunDir, SweepDir
from hydra_zen import store

from lightning_hydra_zen_template.configs.utils.paths import log_dir, output_dir

year_month_day: str = "${now:%Y-%m-%d}"
hour_minute_second: str = "${now:%H-%M-%S}"
task_name: str = "${task_name}"

run_dir: str = os.path.join(log_dir, task_name, "runs", year_month_day, hour_minute_second)
sweep_dir: str = os.path.join(log_dir, task_name, "multiruns", year_month_day, hour_minute_second)
job_file: str = os.path.join(output_dir, f"{task_name}.log")


HydraCfg = HydraConf(
    run=RunDir(run_dir),
    sweep=SweepDir(dir=sweep_dir, subdir="${hydra:job.num}"),
    # Fix from PR https://github.com/facebookresearch/hydra/pull/2242, while there isn't a new release
    job_logging={"handlers": {"file": {"filename": job_file}}},
)


store(HydraCfg)
