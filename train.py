from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel


seed_everything(42)

parser = ArgumentParser()
parser = MNISTDataModule.add_argparse_args(parser)
parser = MNISTModel.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

datamodule = MNISTDataModule.from_argparse_args(args)
model = MNISTModel(**args.__dict__)
# wandb_logger = pl_loggers.WandbLogger()
trainer: Trainer = Trainer.from_argparse_args(
    args,
)

trainer.fit(model, datamodule)
trainer.test(model, datamodule=datamodule)
