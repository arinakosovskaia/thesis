import re
import pandas as pd
from typing import List, Tuple
import torch
# from sklearn.model_selection import train_test_split
import torch.nn as nn
import transformers
from transformers import XLMRobertaModel, XLMRobertaConfig


# inspired by this tutorial:
# https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb#scrollTo=Bo7kpNZnRzoB

def parse_dataset(file_path: str) -> Tuple[List[str], List[int]]:
    comments = []
    classes = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match("__label__(\w+) (.*)", line)
            if match:
                label = match.group(1)
                text = match.group(2)
                text = remove_non_alphanumeric(text)
                text = remove_extra_spaces(text)
                label_num = 0 if label == 'NORMAL' else 1
                comments.append(text)
                classes.append(label_num)

    return comments, classes


def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r'[^\w\s.,!?\'&+*#@%_()-]', '', text)


def remove_extra_spaces(text: str) -> str:
    text = re.sub(r'\s([.,!?])', r'\1', text)
    text = text.replace('( ', '(')
    text = text.strip()
    return text


def process_comments(comments: pd.DataFrame) -> pd.DataFrame:
    comments = comments[~comments['comment'].str.contains('http')]
    comments.loc[:, 'comment'] = comments['comment'].str.rstrip('\n')
    comments.loc[:, 'comment'] = comments['comment'].apply(lambda x: re.sub(r'[^\w\s.,!?\'&+*#@%_()-]', '', x))
    comments.loc[:, 'comment'] = comments['comment'].apply(lambda x: re.sub(r'\s([.,!?])', r'\1', x))
    return comments


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled_data = df.sample(frac=1, random_state=42)
    train_data, temp_data = train_test_split(shuffled_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data


def convert_to_features(example_batch):
    inputs = list(example_batch['comment'])
    features = tokenizer.batch_encode_plus(
        inputs, max_length=512, pad_to_max_length=True
    )
    features["labels"] = example_batch["toxic"]
    return features


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    q = False
    if q:
        texts, labels = parse_dataset("dataset.txt")
        comments1 = pd.DataFrame({'comment': texts, 'toxic': labels})
        comments2 = process_comments(pd.read_csv('labeled.csv'))
        result = pd.concat([comments1, comments2])
        result.to_csv("toxic_comments.csv", index=False)
        train, val, test = split_dataset(result)
    train = pd.read_csv('train.csv')
    val = pd.read_csv('val.csv')
    test = pd.read_csv('test.csv')

    model_name = "sismetanin/xlm_roberta_large-ru-sentiment-rusentiment"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    train_encodings = tokenizer(list(train['comment']), truncation=True, padding=True)
    val_encodings = tokenizer(list(val['comment']), truncation=True, padding=True)
    test_encodings = tokenizer(list(test['comment']), truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, list(train['toxic']))
    val_dataset = CustomDataset(val_encodings, list(val['toxic']))
    test_dataset = CustomDataset(test_encodings, list(test['toxic']))

    dataset_dict = {
        "ethics": train_dataset,
        "toxicity": val_dataset,
        "appropriate": test_dataset
    }


    def convert_to_features(custom_dataset):
        features = {
            "input_ids": custom_dataset.encodings['input_ids'],
            "attention_mask": custom_dataset.encodings['attention_mask'],
            "labels": torch.tensor(custom_dataset.labels),
        }
        return features


    columns_dict = {
        "ethics": ['input_ids', 'attention_mask', 'labels'],
        "toxicity": ['input_ids', 'attention_mask', 'labels'],
        "appropriate": ['input_ids', 'attention_mask', 'labels'],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        features_dict[task_name]["train"] = convert_to_features(dataset)
