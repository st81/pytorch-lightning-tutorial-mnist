from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel
from utils import set_args_by_config_file


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser = MNISTDataModule.add_argparse_args(parser)
    parser = MNISTModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser


def prepare_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args = set_args_by_config_file(args)
    return args


def main(args: Namespace) -> None:
    datamodule = MNISTDataModule.from_argparse_args(args)
    model = MNISTModel(**args.__dict__)
    # wandb_logger = pl_loggers.WandbLogger()
    trainer: Trainer = Trainer.from_argparse_args(
        args,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    seed_everything(42)
    args = prepare_args(prepare_parser())
    main(args)
