import argparse
import os
import pandas as pd
from pytorch_lightning import Trainer
from data import ToxicCommentsDataset
from model import ToxicCommentTagger
from transformers import BertTokenizerFast as BertTokenizer
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Single csv to test")
    parser.add_argument("--data_dir", type=str, help="Test every csv in the folder")
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument(
        "--model_names",
        type=str,
        help="Comma separated names of the folders inside ckpt_dir containing the checkpoints",
    )
    parser.add_argument(
        "--renames",
        type=str,
        default=None,
        help="Comma separated list of names for the column for each model",
    )
    parser.add_argument("--input_column", type=str, default="comment_text")
    parser.add_argument("--target_column", type=str, default="target")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_paths = []
    test_dataloaders = []
    outputs = []

    if args.data_path is None:
        for fname in os.listdir(args.data_dir):
            if fname.startswith("test"):
                data_paths.append(os.path.join(args.data_dir, fname))

        if len(data_paths) == 0:
            raise Exception("Test file must start with 'test'")
    else:
        data_paths.append(args.data_path)

    print(data_paths)

    for data_path in data_paths:
        test_df = pd.read_csv(data_path)[[args.input_column, args.target_column]]
        test_df = test_df.rename(columns={args.input_column: "comment_text"})
        test_df["envs"] = 0
        test_dataset = ToxicCommentsDataset(
            args.target_column, test_df, tokenizer, args.max_length
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )
        test_dataloaders.append(test_dataloader)

    trainer = Trainer(accelerator="gpu", devices=1)

    for i, model_name in enumerate(args.model_names.split(",")):
        # Logic based on wandb's local checkpoint folder structure
        if os.path.isdir(os.path.join(args.ckpt_dir, model_name, "checkpoints")):
            path = f"checkpoints/{os.listdir(os.path.join(args.ckpt_dir, model_name, 'checkpoints'))[0]}"
        else:
            path = "last.ckpt"

        model = ToxicCommentTagger.load_from_checkpoint(
            os.path.join(args.ckpt_dir, model_name, path)
        )
        model.config = "vanilla"

        for j, test_dataloader in enumerate(test_dataloaders):
            result = trainer.test(model, test_dataloader)[0]
            result["model_ckpt"] = model_name
            result["df"] = os.path.basename(data_paths[j])

            if args.renames is not None:
                result["model_name"] = args.renames.split(",")[i]

            outputs.append(result)
            print(
                f"^: {result['df']} | {model_name}"
                + (f" | {result['model_name']}" if args.renames is not None else "")
            )

    if args.save_path is not None:
        pd.DataFrame(outputs).to_csv(args.save_path, index=False)
        print(f"Saved to {args.save_path}.")


if __name__ == "__main__":
    main()
