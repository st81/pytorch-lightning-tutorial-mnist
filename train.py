from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel
from utils import prepare_callbacks, set_args_by_config_file, prepare_loggers


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser = MNISTDataModule.add_argparse_args(parser)
    parser = MNISTModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser


def prepare_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args = set_args_by_config_file(args, is_overrode=True)
    return args


def main(args: Namespace) -> None:
    datamodule = MNISTDataModule.from_argparse_args(args)
    model = MNISTModel(**args.__dict__)
    # wandb_logger = pl_loggers.WandbLogger()
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        default_root_dir="dir",
        callbacks=prepare_callbacks(),
        logger=prepare_loggers(),
    )
    if args.auto_lr_find or args.auto_scale_batch_size is not None:
        trainer.tune(model, datamodule)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    load_dotenv()
    seed_everything(42, workers=True)
    args = prepare_args(prepare_parser())
    main(args)
