from typing import Optional, Union, List
from argparse import ArgumentParser, Namespace
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("MNISTDataModule")
        parser.add_argument("--data_dir", type=str, default="data")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--train_size", type=float, default=0.7)
        return parent_parser

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.train_size = args.train_size

    def prepare_data(self) -> None:
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081))])
        if stage in (None, "fit"):
            dataset = MNIST(
                root=self.data_dir, train=True, download=False, transform=transform,
            )
            num_data = len(dataset)
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [
                    int(num_data * self.train_size),
                    int(num_data * (1.0 - self.train_size)),
                ],
            )
        if stage in (None, "test"):
            self.test_dataset = MNIST(
                root=self.data_dir, train=False, download=False, transform=transform,
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
