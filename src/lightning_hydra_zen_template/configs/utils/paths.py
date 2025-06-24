import os

from rootutils import find_root

root_dir: str = str(find_root(search_from=__file__))
data_dir: str = os.path.join(root_dir, "data")
log_dir: str = os.path.join(root_dir, "logs")
output_dir: str = "${hydra:runtime.output_dir}"
work_dir: str = "${hydra:runtime.cwd}"
