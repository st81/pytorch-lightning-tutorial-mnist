from argparse import Namespace

from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from datamodules.bases import MNISTDataModule
from models.bases import MNISTModel
from utils.argparse import prepare_args, prepare_parser
from utils.utils import prepare_callbacks, prepare_loggers


def main(args: Namespace) -> None:
    datamodule = MNISTDataModule.from_argparse_args(args)
    model = MNISTModel(**args.__dict__)
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        callbacks=prepare_callbacks(args.is_swa),
        logger=prepare_loggers(args.wandb_id),
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
