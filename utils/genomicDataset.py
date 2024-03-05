import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import random_split

class GenomicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        max_length,
        tokenizer,
        use_padding=None,
        add_eos=False,
    ):
        self.file_path = file_path
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer = tokenizer
        self.add_eos = add_eos

        # Load data from the text file
        with open(file_path, "r") as file:
            lines = file.readlines()
            self.data = [line.strip() for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        # Tokenize the input text
        encoding = self.tokenizer(
            x,
            add_special_tokens=True,
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=False,
        )
        seq = encoding["input_ids"]

        # Need to handle eos here
        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)

        seq = torch.LongTensor(seq)

        return seq

def train_test_dataset_split(dataset, train_ratio=0.8, random_seed=None):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset
    
    
    
class GenomicDataset2(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        label_path,
        max_length,
        tokenizer,
        use_padding=None,
        add_eos=False,
        add_special_tokens = True,
    ):
        self.file_path = file_path
        self.label_path = label_path
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.add_special_tokens = add_special_tokens

        # Load data from the text file
        with open(file_path, "r") as file:
            lines = file.readlines()
            self.data = [line.strip() for line in lines]

        # Load labels from the Parquet file
        labels_df = pd.read_parquet(label_path, engine='pyarrow')
        self.labels = labels_df["phenotype"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]

        # Tokenize the input text
        encoding = self.tokenizer(
            x,
            add_special_tokens=self.add_special_tokens,
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=False,
        )
        seq = encoding["input_ids"]

        # Need to handle eos here
        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)
        
        seq = torch.LongTensor(seq)
        label = torch.FloatTensor([label])

        return seq, label

class GenomicDataset3(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        label_path,
        max_length,
        tokenizer,
        use_padding=None,
        add_eos=False,
        add_special_tokens = True,
    ):
        self.file_path = file_path
        self.label_path = label_path
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.add_special_tokens = add_special_tokens

        # Load data from the text file
        with open(file_path, "r") as file:
            lines = file.readlines()
            self.data = [line.strip() for line in lines]

        # Load labels from the Parquet file
        labels_df = pd.read_parquet(label_path, engine='pyarrow')
        self.labels = labels_df["phenotype"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
       
        # Tokenize the input text
        encoding = self.tokenizer(
            x,
            add_special_tokens=self.add_special_tokens,
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=False,
        )
        seq = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Need to handle eos here
        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)

        seq = torch.LongTensor(seq)
        label = torch.FloatTensor([label])
        attention_mask = torch.LongTensor(attention_mask)
        item = {"input_ids": seq, "label": label, "attention_mask": attention_mask}

        return item

class GenomicDataset4(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        label_path,
        max_length,
        tokenizer,
        use_padding=None,
        add_eos=False,
        add_special_tokens = True,
    ):
        self.file_path = file_path
        self.label_path = label_path
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.add_special_tokens = add_special_tokens

        # Load data from the text file
        with open(file_path, "r") as file:
            lines = file.readlines()
            self.data = [line.strip() for line in lines]

        # Load labels from the Parquet file
        labels_df = pd.read_parquet(label_path, engine='pyarrow')
        self.labels = labels_df["phenotype"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
       
        # Tokenize the input text
        encoding = self.tokenizer(
            x,
            add_special_tokens=self.add_special_tokens,
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=False,
        )
        seq = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Need to handle eos here
        if self.add_eos:
            seq.append(self.tokenizer.sep_token_id)

        seq = torch.LongTensor(seq)

        label = torch.FloatTensor([label])
        
        attention_mask = torch.LongTensor(attention_mask)

        item = {"input_ids": seq, "label": label, "attention_mask": attention_mask}

        return seq, label, attention_mask