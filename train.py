from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel
from models.bases import LitMNIST


seed_everything(42)

parser = ArgumentParser()
parser = MNISTDataModule.add_argparse_args(parser)
parser = MNISTModel.add_argparse_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

model = MNISTModel(args)
# model = LitMNIST()
datamodule = MNISTDataModule(args)
trainer = Trainer.from_argparse_args(args)

trainer.fit(model, datamodule)
trainer.test(model, datamodule=datamodule)
