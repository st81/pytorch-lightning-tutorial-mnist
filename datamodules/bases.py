import os
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 256,
        train_size: float = 0.7,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081))])
        if stage in (None, "fit"):
            dataset = MNIST(
                root=self.data_dir,
                train=True,
                download=False,
                transform=transform,
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
                root=self.data_dir,
                train=False,
                download=False,
                transform=transform,
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
