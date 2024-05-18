import os
import pandas as pd
import re
import time
from typing import List, Dict, Tuple
from perspective import Client, Attributes
from tqdm import tqdm

API_KEY = ""
client = Client(token=API_KEY)


def text_length(text: str) -> bool:
    return 64 <= len(text) <= 1024


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df['prompt'] = df['prompt'].str.strip().str.lstrip('\'')
    df['prompt'] = df['prompt'].apply(lambda x: re.sub(r'^[^\w\s]+', '', x))
    df['prompt'] = df['prompt'].apply(lambda x: re.sub(r'(?<=\W)(?=[^\s\W])', ' ', x))
    df['prompt'] = df['prompt'].str.strip()
    df['prompt'] = df['prompt'].apply(lambda x: x.capitalize())
    return df


# for twtexts.txt
def clean_text(text: str) -> str:
    text = re.sub(r'\bRT\b', '', text)  # delete RT
    text = re.sub(r'@\w+', '', text)  # delete @username
    text = re.sub(r'\s+', ' ', text).strip()  # delete extra spaces
    text = re.sub(r'^:\s+', '', text)  # delete ": " в in the beginning of line
    return text


# for vktexts.txt
def remove_id_mentions(text: str) -> str:
    text = re.sub(r'\[id\d+\|[^\]]+\]', '', text)  # delete [idXXXXXX|Name]
    text = re.sub(r'\s+', ' ', text).strip()  # delete extra spaces
    text = re.sub(r'^,\s+', '', text)  # delete ", " в in the beginning of line
    return text


def get_prompt(sentence) -> Tuple[str, bool]:
    words = sentence.split()
    midpoint_index = len(words) // 2
    prompt = ' '.join(words[:midpoint_index])
    length_check = len(prompt) <= 128
    return prompt, length_check


def statistics(df: pd.DataFrame) -> pd.Series:
    prompt_statictics = pd.cut(df['prompt toxicity score'], bins=[0, 0.25, 0.5, 0.75, 1],
                               include_lowest=True).value_counts()
    sentence_statistics = pd.cut(df['sentence toxicity score'], bins=[0, 0.25, 0.5, 0.75, 1],
                                 include_lowest=True).value_counts()
    return prompt_statictics, sentence_statistics


def is_russian_text(text: str) -> bool:
    allowed_chars = ' абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,;:!?\"\'-»«'
    return all(char in allowed_chars for char in text)


def toxicity(text: str) -> float:
    time.sleep(1)
    try:
        response = client.analyze(text=text, attributes=[Attributes.TOXICITY])
        return response['TOXICITY'] / 100
    except Exception as e:
        print(f"Error '{e}' in text: '{text}'")
        return None


def split_into_sentences(text: str) -> List[str]:
    # Splitting by "DataBaseItem: number\n" then by sentence delimiters
    sentences = []
    for items in text.split('DataBaseItem'):
        cur_line = '\n'.join(items.split('\n')[1:])
        sentences += re.split(r'(?<=[.!?])\s+', cur_line)
    return sentences


class TextProcessor:
    def __init__(self, max_group_sizes: dict):
        self.data = []
        self.max_group_sizes = max_group_sizes
        self.group_sizes = {group: 0 for group in max_group_sizes}  # Groups based on toxicity scores
        self.last_time = time.time()
        self.log_iter = 0

    def logging(self):
        print(self.group_sizes)
        df = self.groups_to_dataframe()
        cur_name = "data" + str(self.log_iter) + ".csv"
        df.to_csv(cur_name, index=False, encoding='utf-8')
        self.log_iter = self.log_iter + 1

    def read_and_process_files_twitter(self, file_paths: List[str]):
        last_time = time.time()
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            russian_df = df[df['language'] == 'Russian']
            content = russian_df['content'].values
            for text in tqdm(content):
                if time.time() - last_time >= 1800:
                    self.logging()
                    last_time = time.time()
                text = clean_text(text)
                if text_length(text) and is_russian_text(text):
                    self.process_sentence(text)

    def read_and_process_files_taiga(self, file_paths: List[str]):
        for file_path in file_paths:
            name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                last_time = time.time()
                sentences = split_into_sentences(text)
                sentences = sentences[340191:]
                for sentence in tqdm(sentences):
                    if time.time() - last_time >= 1800:
                        self.logging()
                        last_time = time.time()
                    if text_length(sentence):
                        if name == 'vktexts.txt':
                            sentence = remove_id_mentions(sentence)
                        elif name == 'twtexts.txt':
                            sentence = clean_text(sentence)
                            if not is_russian_text(sentence):
                                continue
                        self.process_sentence(sentence)

    def process_sentence(self, sentence: str):
        toxicity_score = toxicity(sentence)
        if toxicity_score is None:
            return
        group_number = self.get_group_number(toxicity_score)
        if group_number and self.group_sizes[group_number] < self.max_group_sizes[group_number]:
            prompt, length_check = get_prompt(sentence)
            if length_check:
                prompt_toxicity_score = toxicity(prompt)
                if prompt_toxicity_score is None:
                    return
                self.data.append((prompt, prompt_toxicity_score, toxicity_score))
                self.group_sizes[group_number] += 1

    @staticmethod
    def get_group_number(score: float) -> int:
        if score < 0.25:
            return 1
        elif 0.25 < score < 0.5:
            return 2
        elif 0.5 < score < 0.75:
            return 3
        elif 0.75 < score < 1:
            return 4

    def groups_to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data, columns=['prompt', 'prompt toxicity score', 'sentence toxicity score'])
        return df
