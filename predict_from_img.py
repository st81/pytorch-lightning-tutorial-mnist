import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import polars
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel

seed_everything(42, workers=True)


class MNISTModelNative(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        # architecture attributes
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)


class MNISTModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parenet_parser: ArgumentParser) -> ArgumentParser:
        parser = parenet_parser.add_argument_group("MNISTModel")
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--hidden_size", type=int, default=64)
        return parenet_parser

    def __init__(
        self,
        learning_rate: float,
        hidden_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        # architecture attributes
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.num_classes),
        )

        # metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def shared_step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log(f"loss_{stage}", loss, on_step=False, on_epoch=True)
        preds = torch.argmax(logits, dim=1)
        if stage == "train":
            self.train_acc(preds, y)
        elif stage == "val":
            self.val_acc(preds, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("acc_train", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "val")
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log("acc_val", self.val_acc.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", self.test_acc.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)


class MNISTDataset(Dataset):
    def __init__(
        self, img_paths: List[str], targets: Optional[List[int]] = None, transform: Optional[Callable] = None
    ) -> None:
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx: int):
        img = Image.open(self.img_paths[idx])

        if self.transform is not None:
            img = self.transform(img)

        if self.targets is not None:
            target = self.targets[idx]
            return img, target
        else:
            return img

    def __len__(self) -> int:
        return len(self.img_paths)

DEVICE = "cpu"

data_dir = "data/mnist_imgs"
df = polars.read_csv("data/test.csv")

img_paths = df["img_path"].to_list()
targets = df["target"].to_list()
# num_repeat = 3000
num_repeat = 5000
img_paths = img_paths * num_repeat
targets = targets * num_repeat

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081))])
test_dataset = MNISTDataset(img_paths, targets, transform)
dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
# datamodule = MNISTDataModule(img_paths=img_paths, targets=targets)
# datamodule.setup("test")
checkpoint = torch.load("output/2024-12-18-145359/checkpoints/epoch2-acc_val0.89.ckpt")
print(checkpoint.keys())
hparams = checkpoint["hyper_parameters"]
# model = MNISTModel.load_from_checkpoint("output/2024-12-18-145359/checkpoints/epoch2-acc_val0.89.ckpt")
# model.model.eval()
model = MNISTModelNative(hidden_size=hparams["hidden_size"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.to(DEVICE)

all_preds = []
for batch in tqdm(dataloader):
    x, y = batch
    x = x.to(DEVICE)
    logits = model(x)
    preds = torch.argmax(logits, dim=1)
    all_preds.append(preds.detach().cpu().numpy())
all_preds = np.concatenate(all_preds)
print(all_preds.shape)
# compute accuracy by sklearn

y_true = df["target"].to_list() * num_repeat
acc = accuracy_score(y_true, all_preds)

print(f"Accuracy: {acc}")
