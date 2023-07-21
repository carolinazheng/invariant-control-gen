# An Invariant Learning Characterization of Controlled Text Generation
This repo contains source code for the [paper](https://arxiv.org/abs/2306.00198):

```
An Invariant Learning Characterization of Controlled Text Generation
Carolina Zheng*, Claudia Shi*, Keyon Vafa, Amir Feder, David M. Blei
ACL 2023
```

## Requirements
Our Python version is `3.10`. The required packages can be installed inside a virtual environment via

`pip install -r requirements.txt`

To run `civilcomments_preprocess.py`, first you need to run 

`python -m spacy download en_core_web_sm`

## Datasets
To preprocess CivilComments for training (subsampling and creating environments used in the paper), run the script

`python datasets/civilcomments_preprocess.py --data_dir [DATA_DIR] --save_dir [SAVE_DIR] --n_envs [N_ENVS] --n_total 28000 --n_test 4200 --balanced 0 --load_evian 0 --evian_model_dir [MODEL_DIR] --seed 42`

- `--data_dir` is the directory with the CivilComments data (download from [here](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data))
- `--save_dir` is the directory to save the generated splits (train/valid/test)
- `--n_envs` is the number of environments (supports 2 or 3)
- `--n_total` is the number of CivilComments examples to subset
- `--n_test` is the number of examples in valid/test (for each)
- `--balanced` is 1 to sample toxic/nontoxic examples equally, 0 to use the data distribution (0 in the paper)
- `--load_evian` is 1 to load previously trained EviaN classifiers from `evian_model_dir`, 0 to train the models in the script
- `--evian_model_dir` is the directory to either load or save the EviaN classifiers to
- `--seed` is the random seed value

To generate the RealToxicityPrompts dataset, run the script

`python datasets/generate_rtp.py --api_key [KEY] --save_dir [SAVE_DIR] --data_dir [DATA_DIR] --n_prompts 200 --n_continuations 10 --seed 42`

- `--api_key` is your OpenAI API key
- `--save_dir` is the directory to save the generated dataset as `rtp.csv`
- `--data_dir` is the directory with the RealToxicityPrompts data (download from [here](https://allenai.org/data/real-toxicity-prompts))
- `--n_prompts` is the total number of prompts to sample
- `--n_continuations` is the number of times to sample GPT-3 per prompt
- `--seed` is the random seed value

To generate the Personification dataset, run the script

`python datasets/generate_personification.py --api_key [KEY] --save_dir [SAVE_DIR] --winogenerated_path [WINO_PATH] --n_continuations 10 --n_jobs 25 --seed 42`

- `--api_key` is your OpenAI API key
- `--save_dir` is the directory to save the generated dataset as `all-professions.csv`
- `--winogenerated_path` is the path to the WinoBias professions file (download from [here](https://github.com/anthropics/evals/blob/main/winogenerated/winogenerated_examples.jsonl))
- `--n_continuations` is the number of times to sample GPT-3 per prompt
- `--n_jobs` is the number of WinoBias professions to use
- `--seed` is the random seed value

## Usage
To train the model, run the script

`python train.py --data_dir [DATA_DIR] --log_dir [LOG_DIR] --save_dir [SAVE_DIR] --valid_in_train [0 or 1] --config [CONFIG] --env_name [ENV_NAME] --beta [BETA] --warmup_beta [WARMUP_BETA]  --target target --early_stop 0 --max_length 256 --batch_size 120 --n_epochs 4 --lr 1e-5 --lr_schedule warmup_linear_decay --n_gpus 4 --n_workers 1 --debug 0 --seed 42`

- `--data_dir` is the data directory with train.csv, test.csv, and valid.csv (unless valid_in_train)
- `--log_dir` is the directory to save Tensorboard logs
- `--save_dir` is the directory to save the best model checkpoint
- `--valid_in_train` is 1 if using leave-one-environment-out-validation (i.e., train data was preprocessed to create 3 environments), otherwise 0
- `--config` is the invariance algorithm, options are `vanilla` (ERM), `v-rex`, `mmd`, or `coral`
- `--env_name` is the name of the environment column, e.g., `evian_metadata`
- `--beta` is the invariance regularizer strength value
- `--warmup_beta` is the proportion of training steps to linearly scale beta from 0 to 1 (0.1 for V-REx, 0 for other algorithms)
- `--target` is the name of the column with y labels
- `--early_stop` is the early stopping patience in epochs, or 0 for no early stopping
- `--max_length` is the max number of tokens per example
- `--batch_size` is the per-GPU batch size
- `--n_epochs` is the number of training epochs
- `--lr` is the learning rate
- `--lr_schedule` is the learning rate schedule, options are `constant` or `warmup_linear_decay`
- `--n_gpus` is the number of GPUs for Pytorch Lightning DDP
- `--n_workers` is the number of workers to create the dataloaders
- `--debug` is 1 for debug mode (no saving or logging), otherwise 0
- `--seed` is the random seed value

You can also use [Weights and Biases](https://wandb.ai/) for logging and checkpointing by specifying the following arguments instead of `--log_dir` and `--save_dir`:

`--wandb 1 --wandb_run_name [RUN_NAME] --wandb_project [PROJECT] --wandb_entity [ENTITY]`

To test the model on one or more test CSVs (runs on a single GPU), run the script

`python test.py --data_path [CSV_PATH] --ckpt_dir [CKPT_DIR] --save_path [SAVE_PATH] --model_names [MODEL_NAMES] --renames [RENAMES] --input_column comment_text --target_column target --max_length 256 --batch_size 32 --n_workers 1`

- `--data_path` is the path to a single test CSV (alternatively, specify `--data_dir` as the directory containing multiple test CSVs that start with "test")
- `--ckpt_dir` is the `save_dir` from training
- `--save_path` is the path to save the output results CSV
- `--model_names` is a comma-separated list of directories within `ckpt_dir` for the individual models
- `--renames` is a comma-separated list of strings to rename each model in the results CSV
- `--input_column` is the name of the column containing the examples
- `--target_column` is the name of the column containing the labels
- `--max_length` is the max number of tokens per example
- `--batch_size` is the batch size
- `--n_workers` is the number of workers to create the dataloaders

## Bibtex Citation
```
@inproceedings{zheng-etal-2023-invariant,
    title = "An Invariant Learning Characterization of Controlled Text Generation",
    author = "Zheng, Carolina and Shi, Claudia and Vafa, Keyon and Feder, Amir and Blei, David",
    booktitle = "Association for Computational Linguistics",
    year = "2023",
}
```
