from argparse import Namespace
from pathlib import Path
from typing import List, Union
import yaml
from datetime import datetime

from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers import WandbLogger


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


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


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print(f"Training is starting {trainer.__dict__}")


def prepare_callbacks() -> List[Callback]:
    return [
        ModelCheckpoint(
            dirpath=f"output/{now()}/checkpoints",
            filename="epoch{epoch}-acc_val{acc_val:.2f}",
            monitor="acc_val",
            mode="max",
            verbose=True,
            auto_insert_metric_name=False,
        ),
        # StochasticWeightAveraging(swa_epoch_start=5),
    ]


def prepare_loggers() -> List[LightningLoggerBase]:
    return [
        WandbLogger(
            name=now(),
            project="pytorch-lightning-tutorial-mnist",
            id="1x6nqfou",
        )
    ]
