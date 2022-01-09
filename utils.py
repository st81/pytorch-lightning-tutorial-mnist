from argparse import Namespace
from pathlib import Path
from typing import Union
import yaml


def set_args_by_config_file(
    args: Namespace, path: Union[Path, str] = Path("configs/default.yaml")
) -> Namespace:
    with open(Path(path), "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    for k, v in config.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args
