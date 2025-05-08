import os

from hydra_zen import make_config
from rootutils import find_root

root_dir = str(find_root(search_from=__file__))
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")
output_dir = "${hydra:runtime.output_dir}"
work_dir = "${hydra:runtime.cwd}"

PathsCfg = make_config(
    root_dir=root_dir,
    data_dir=data_dir,
    log_dir=log_dir,
    output_dir=output_dir,
    work_dir=work_dir,
)
