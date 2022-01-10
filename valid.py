import pytorch_lightning as pl

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel


datamodule = MNISTDataModule()
datamodule.setup("fit")
model = MNISTModel.load_from_checkpoint(
    "lightning_logs/version_29/checkpoints/epoch=2-step=494.ckpt"
)
trainer = pl.Trainer()
trainer.validate(model, datamodule=datamodule)