from typing import List, Optional
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


def prepare_callbacks(is_swa: bool = False) -> List[Callback]:
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=f"output/{now()}/checkpoints",
            filename="epoch{epoch}-acc_val{acc_val:.2f}",
            monitor="acc_val",
            mode="max",
            verbose=True,
            auto_insert_metric_name=False,
        )
    )
    if is_swa:
        callbacks.append(StochasticWeightAveraging())
    return callbacks


def prepare_loggers(wandb_id: Optional[str] = None) -> List[LightningLoggerBase]:
    return [
        WandbLogger(
            name=now(),
            project="pytorch-lightning-tutorial-mnist",
            id=wandb_id if wandb_id != "" else None,
        )
    ]
