from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import yaml
from pytorch_lightning import Trainer

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser = MNISTDataModule.add_argparse_args(parser)
    parser = MNISTModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser


def prepare_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args = set_args_by_config_file(args, is_overrode=True)
    return args


def set_args_by_config_file(
    args: Namespace,
    is_overrode=False,
    path: Union[Path, str] = Path("configs/default.yaml"),
) -> Namespace:
    with open(Path(path), "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    for k, v in config.items():
        if is_overrode:
            setattr(args, k, v)
        elif getattr(args, k) is None:
            setattr(args, k, v)
    return args
