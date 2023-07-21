import argparse
import numpy as np
import os
import pandas as pd
import random
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast as BertTokenizer

from data import ToxicCommentsDataset
from model import ToxicCommentTagger


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "--valid_in_train",
        type=str_to_bool,
        default=False,
        help="If true, create validation data using the third environment in the training data.",
    )
    parser.add_argument("--config", type=str, default="vanilla")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--warmup_beta", type=float, default=0.0)
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--early_stop", type=str_to_bool, default=False)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear_decay")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--wandb", type=str_to_bool, default=False)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("*****************************")
    print(args)
    print("*****************************")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataloaders = {}

    def make_set_dataloader(df, split, tokenizer):
        dataset = ToxicCommentsDataset(args.target, df, tokenizer, args.max_length)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.n_workers,
        )

    for split in ["train", "valid", "test"]:
        if args.valid_in_train and split == "valid":
            continue

        df = pd.read_csv(os.path.join(args.data_dir, f"{split}.csv"))

        # Set envs column based on env_name
        if args.env_name is None or split == "test":
            df["envs"] = 0
        else:
            df.rename(columns={f"env_{args.env_name}": "envs"}, inplace=True)

        if args.valid_in_train and split == "train":
            if args.env_name is None:
                raise Exception("Must specify env_name when using valid_in_train=True")
            if df["envs"].nunique() != 3:
                raise Exception(
                    "Must have 3 environments when using valid_in_train=True"
                )

            valid_df = df[df["envs"] == 2].reset_index(drop=True)
            valid_df["envs"] = 0
            make_set_dataloader(valid_df, "valid", tokenizer)
            train_df = df[df["envs"] < 2].reset_index(drop=True)
            make_set_dataloader(train_df, "train", tokenizer)
        else:
            make_set_dataloader(df, split, tokenizer)

    model = ToxicCommentTagger(args)

    callbacks = []

    if not args.debug:
        if args.wandb:
            logger = WandbLogger(
                name=args.wandb_run_name,
                projects=args.wandb_project,
                entity=args.wandb_entity,
                log_model=False,
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                mode="min",
            )
        else:
            if args.log_dir is None:
                raise Exception(
                    "Please specify --log_dir to log this run with Tensorboard."
                )
            if args.save_dir is None:
                raise Exception("Please specify --save_dir to save this model.")

            n_envs = (
                1 if args.config == "vanilla" else (3 if args.valid_in_train else 2)
            )

            run_name = f"{args.config}_{args.env_name}_{n_envs}e_b{args.beta}"
            logger = TensorBoardLogger(
                args.log_dir,
                name=run_name,
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(args.save_dir, run_name),
                monitor="val/loss",
                save_last=True,
            )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        callbacks.append(checkpoint_callback)

    if args.early_stop:
        early_stop_callback = EarlyStopping(
            monitor="val/loss", min_delta=0.01, patience=5, verbose=False, mode="min"
        )
        callbacks.append(early_stop_callback)

    trainer = Trainer(
        accelerator="gpu",
        devices=args.n_gpus,
        strategy="ddp",
        logger=logger if not args.debug else None,
        max_epochs=args.n_epochs,
        callbacks=callbacks,
        log_every_n_steps=25,
        val_check_interval=1.0,
    )

    trainer.validate(model, dataloaders["valid"])
    trainer.fit(model, dataloaders["train"], dataloaders["valid"])

    # With pl, test metrics slightly wrong when used with ddp and >1 devices, need separate script.
    if args.debug:
        trainer.test(model, dataloaders["test"])
    else:
        trainer.test(
            dataloaders=dataloaders["test"],
            ckpt_path=checkpoint_callback.best_model_path,
        )


if __name__ == "__main__":
    main()
