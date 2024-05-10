import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from implicit_model import ProcessText
from sklearn.model_selection import train_test_split

np.random.seed(42)

def augment_dataset(df, df1=None, size=1000, size1=50):
    """
    This function increases the given dataset by 'size' elements by combining texts with label=1 and label=0.
    If df1 is provided, increases df1 by 'size1' elements also.
    """
    texts_label_0 = df[df['label'] == 0]['text'].tolist()
    texts_label_1 = df[df['label'] == 1]['text'].tolist()
    new_rows = []

    for _ in range(size):
        text_0 = random.choice(texts_label_0)
        text_1 = random.choice(texts_label_1)
        new_text = text_1 + " " + text_0
        new_rows.append({'text': new_text, 'label': 1})

    texts_label_0 = df1[df1['label'] == 0]['text'].tolist()
    texts_label_1 = df1[df1['label'] == 1]['text'].tolist()
    if df1:
        for _ in range(size1):
            text_0 = random.choice(texts_label_0)
            text_1 = random.choice(texts_label_1)
            new_text = text_1 + " " + text_0
            new_rows.append({'text': new_text, 'label': 1})

    new_df = pd.DataFrame(new_rows)
    df_augmented = pd.concat([df, new_df], ignore_index=True)
    return df_augmented

def load_and_preprocess(file_path, rename_mapping=None):
    df = pd.read_csv(file_path)
    if rename_mapping:
        df.rename(columns=rename_mapping, inplace=True)
    return df

if __name__ == "__main__":
    train_toxigen = load_and_preprocess('train_toxigen.csv')
    test_toxigen = load_and_preprocess('test_toxigen.csv')
    explicit = load_and_preprocess('labeled.csv', rename_mapping={'toxic': 'label', 'comment': 'text'})
    train_data, test_data = train_test_split(explicit, test_size=0.5, random_state=42, stratify=explicit['label'])

    adversarial = load_and_preprocess('adversarial.csv')
    final_df = pd.concat([adversarial[['text', 'label']], train_toxigen[['text', 'label']]], axis=0)
    df_augmented = augment_dataset(final_df, adversarial, 1000, 50)

    final_train_df = pd.concat([df_augmented[['text', 'label']], train_data[['text', 'label']]], axis=0)
    final_train_df = final_train_df.sample(frac=1).reset_index(drop=True)

    train = final_df.sample(frac=1).reset_index(drop=True)
    test = pd.concat([test_toxigen[['text', 'label']], test_data[['text', 'label']]], axis=0)
    test = test.sample(frac=1).reset_index(drop=True)

    processor = ProcessText()
    df_implicit = DatasetDict({"train": Dataset.from_pandas(train), 
                               "test": Dataset.from_pandas(test)
                               }
                               )

    dataset_processed_implicit = processor.process(df_implicit, num_type=1)
    processor.train(dataset_processed_implicit)
    state_dict = processor.model.state_dict()
    file_path = 'implicit_model.pth'
    torch.save(state_dict, file_path)

