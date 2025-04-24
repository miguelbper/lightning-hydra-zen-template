import logging
import os

from hydra.conf import HydraConf, RunDir, SweepDir
from hydra.experimental.callback import Callback
from hydra_zen import builds, make_config
from omegaconf import DictConfig, OmegaConf

from src.configs.paths import log_dir, output_dir

log = logging.getLogger(__name__)

run_dir = os.path.join(log_dir, "hydra", "runs", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
sweep_dir = os.path.join(log_dir, "hydra", "multiruns", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")
job_file = os.path.join(output_dir, ".log")


class PrintConfigCallback(Callback):
    def on_run_start(self, config: DictConfig, config_name: str | None) -> None:
        # if "hydra" in config:
        #     config = copy.copy(config)
        #     with flag_override(config, ["struct", "readonly"], [False, False]):
        #         config.pop("hydra")
        log.info("Printing composed config")
        print(OmegaConf.to_yaml(config))


HydraCfg = make_config(
    hydra_defaults=[
        {"override hydra/job_logging": "colorlog"},
        {"override hydra/hydra_logging": "colorlog"},
        "_self_",
    ],
    hydra=HydraConf(
        run=RunDir(run_dir),
        sweep=SweepDir(dir=sweep_dir, subdir="${hydra:job.num}"),
        callbacks={"print_config": builds(PrintConfigCallback)},
        # Fix from PR https://github.com/facebookresearch/hydra/pull/2242, while there isn't a new release
        job_logging={"handlers": {"file": {"filename": job_file}}},
    ),
)
