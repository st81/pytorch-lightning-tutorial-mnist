import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel

seed_everything(42, workers=True)

datamodule = MNISTDataModule()
datamodule.setup("fit")
# TODO: Enable to pass checkpoint path as an argument
model = MNISTModel.load_from_checkpoint(
    "lightning_logs/version_29/checkpoints/epoch=2-step=494.ckpt"
)
trainer = pl.Trainer()
trainer.validate(model, datamodule=datamodule)
