from hydra_zen import store, zen

from lightning_hydra_zen_template.configs import EvalCfg
from lightning_hydra_zen_template.funcs.eval import evaluate
from lightning_hydra_zen_template.utils.logging import print_config


def main() -> None:
    """Main entry point for the evaluation script.

    Sets up Hydra configuration and runs the evaluation task with the
    specified configuration.
    """
    store(EvalCfg, name="config")
    store.add_to_hydra_store()
    task_fn = zen(evaluate, pre_call=print_config)
    task_fn.hydra_main(config_path=None, config_name="config", version_base="1.3")


if __name__ == "__main__":
    main()
