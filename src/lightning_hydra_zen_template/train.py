from hydra_zen import store, zen

from lightning_hydra_zen_template.configs import TrainCfg
from lightning_hydra_zen_template.funcs.train import seed_fn, train
from lightning_hydra_zen_template.utils.logging import log_git_status, log_python_env, print_config


def main() -> None:
    """Main entry point for the training script.

    Sets up Hydra configuration and runs the training task with the
    specified configuration.
    """
    store(TrainCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(train, pre_call=[log_git_status, log_python_env, zen(seed_fn), print_config])
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
