import os
from time import time
from typing import List, Dict, Tuple
from perspective import Client, Attributes
from tqdm import tqdm, trange
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time
import pandas as pd

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found in environment variables")
client = Client(token=API_KEY)


class ToxicityAnalyzer:
    def __init__(self, api_key=API_KEY):
        self.client = Client(token=api_key)

    def analyze_toxicity(self, text: str) -> float:
        try:
            time.sleep(1)
            response = self.client.analyze(text=text, attributes=[Attributes.TOXICITY])
            return response['TOXICITY'] / 100
        except Exception as e:
            print(f"Error '{e}' in text: '{text}'")
            return None

    def prospective_score(self, prompts, text_cont=1):
        """
        Analyze toxicity for a set of texts and optionally their continuations.

        :param prompts: DataFrame containing 'text' and optionally 'continuation'.
        :param text_cont: Flag to determine whether to process continuations.
        :return: List of toxicity scores for continuations and texts, or just texts.
        """
        continuations, texts = [], []
        if text_cont:
            for continuation, text in tqdm(zip(prompts['continuation'], prompts['text']), total=len(prompts)):
                toxicity_continuation = self.analyze_toxicity(continuation)
                toxicity_text = self.analyze_toxicity(text)
                continuations.append(toxicity_continuation)
                texts.append(toxicity_text)
            return continuations, texts
        else:
            for text in tqdm(prompts['text']):
                toxicity_text = self.analyze_toxicity(text)
                texts.append(toxicity_text)
            return texts


class PipelineModel:
    def __init__(self, model_path="citizenlab/distilbert-base-multilingual-cased-toxicity",
                 label_name='toxic'):
        """
        Initializes the model and tokenizer.
        """
        self.toxicity_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
        self.label_name = label_name  # name of positive class

    def predict(self, text: str) -> int:
        """
        Predicts whether the text is toxic or not.

        :param text: Text to be classified.
        :return: 1 if text is toxic, 0 otherwise.
        """
        results = self.toxicity_classifier(text)
        if results[0]['label'] == self.label_name:
            return 1
        return 0


class BertClassifier:
    def __init__(self, model_name='SkolkovoInstitute/russian_toxicity_classifier'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def predict(self, text: str, threshold=0.5, need_score=True) -> float:
        """
        Predicts the toxicity score of the given text and can also return
        a binary indication whether the text exceeds a certain toxicity threshold.

        :param text: Input text to classify.
        :param threshold: Threshold for deciding toxic vs non-toxic.
        :param need_score: If True, returns the toxicity score, otherwise returns boolean.
        :return: Toxicity score or boolean classification based on the threshold.
        """
        batch = self.tokenizer.encode(text, return_tensors='pt')
        score = torch.nn.functional.softmax(self.model(batch).logits, dim=1)[:, 1]
        if need_score:
            return score.item()
        return score.item() > threshold


