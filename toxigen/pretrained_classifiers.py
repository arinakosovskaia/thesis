from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
class HateSpeechClassifier():
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()

    def __call__(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def from_text(self, text):
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        logits = self.__call__(input_ids, attention_mask).logits
        return 100 * float(torch.softmax(logits, dim=1)[:, 1].detach().numpy())

class HateBERT(HateSpeechClassifier):
    def __init__(self, model_path='s-nlp/russian_toxicity_classifier'):
        """
        HateBERT files: https://huggingface.co/s-nlp/russian_toxicity_classifier
        """
        super(HateBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()

