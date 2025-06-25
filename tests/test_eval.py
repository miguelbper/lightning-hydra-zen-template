import subprocess
import sys

from rootutils import find_root


def test_main(ckpt_path: str) -> None:
    eval_script = find_root() / "src" / "lightning_hydra_zen_template" / "eval.py"
    subprocess.run([sys.executable, str(eval_script), f"ckpt_path={ckpt_path}"], check=True)
