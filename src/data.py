from pandas import DataFrame
from torch import FloatTensor
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer


class ToxicCommentsDataset(Dataset):
    def __init__(
        self,
        target: str,
        data: DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.target = target
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[self.target]
        encoding = self.tokenizer(
            comment_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        envs = data_row["envs"]

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=FloatTensor([labels]),
            envs=FloatTensor([envs]),
        )
