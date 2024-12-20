import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel

seed_everything(42, workers=True)

datamodule = MNISTDataModule()
datamodule.setup("test")
# TODO: Enable to pass checkpoint path as an argument
model = MNISTModel.load_from_checkpoint(
    "output/2024-12-18-145359/checkpoints/epoch2-acc_val0.89.ckpt"
)
trainer = pl.Trainer()
trainer.test(model, datamodule=datamodule)
