from argparse import ArgumentParser
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


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
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def shared_step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log(f"Loss/{stage}", loss, on_step=False, on_epoch=True)
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
        self.log("Accuracy/train", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "val")
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log("Accuracy/val", self.val_acc.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("Test/loss", loss, on_step=False, on_epoch=True)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log("Test/Accuracy", self.test_acc.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int
    ):
        optimizer.zero_grad(set_to_none=True)


class LitMNIST(pl.LightningModule):
    def __init__(self, data_dir="data", hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307, ), (0.3081, )),
        # ])

        # Define PyTorch model
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

        # metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, y)
        acc = self.val_acc(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
