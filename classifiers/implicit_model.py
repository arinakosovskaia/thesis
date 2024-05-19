import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from transformers import Adafactor, AdafactorSchedule, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


class ProcessText:
    def __init__(self, model_name='s-nlp/russian_toxicity_classifier', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.classifier=nn.Linear(768, 2)
        self.model.dropout = nn.Dropout(p=0.2, inplace=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.f1_metric = load_metric('f1', trust_remote_code=True)
        self.precision_metric = load_metric('precision', trust_remote_code=True)
        self.recall_metric = load_metric('recall', trust_remote_code=True)

    def remove_nan(self, example):
        if example['query'].endswith(' nan'):
            example['query'] = example['query'][:-4]
        return example

    def check_length_type0(self, example):
        tokens = self.tokenizer(example['query'])['input_ids'] + self.tokenizer(example['response'])['input_ids']
        return len(tokens) <= 256

    def check_length_type1(self, example):
        tokens = self.tokenizer(example['text'])['input_ids']
        return len(tokens) <= 256

    def preprocess_datasets(self, dataset_dict, num_type=0):
        #dataset_dict = dataset_dict.remove_columns("score")
        for split in dataset_dict.keys():
            if num_type == 0:
              dataset_dict[split] = dataset_dict[split].map(self.remove_nan)
              dataset_dict[split] = dataset_dict[split].filter(self.check_length_type0)
            elif num_type == 1:
              dataset_dict[split] = dataset_dict[split].filter(self.check_length_type1)
        return dataset_dict

    def tokenize_data_type0(self, example):
        return self.tokenizer(example['query'], example['response'],
                              truncation=True, padding="max_length",
                              max_length=256, return_tensors='pt')

    def tokenize_data_type1(self, example):
        return self.tokenizer(example['text'],
                              truncation=True, padding="max_length",
                              max_length=256)

    def convert_labels(self, batch):
        batch['labels'] = [int(label) for label in batch['label']]
        return batch

    def predict_proba(self, dataloader):
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                true_labels = batch['labels']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                outputs = self.model(**model_inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(true_labels.tolist())
        return all_probs, all_labels


    def evaluate(self, test_dataset):
            test_dataset = test_dataset.map(self.tokenize_data, batched=True)
            test_dataset = test_dataset.map(self.convert_labels, batched=True)
            test_dataset = test_dataset.remove_columns(['label'])
            results = self.trainer.evaluate(test_dataset)
            return results

    def process(self, dataset, num_type=0):
        dataset = self.preprocess_datasets(dataset, num_type)
        if num_type == 0:
          dataset = dataset.map(self.tokenize_data_type0, batched=True)
        else:
          dataset = dataset.map(self.tokenize_data_type1, batched=True)
        dataset = dataset.map(self.convert_labels, batched=True)
        self.dataset = dataset.remove_columns(['label'])
        return self.dataset

    def compute_metrics(self, eval_preds, threshold=0.5):
        logits, labels = eval_preds
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
        predictions = (probs[:, 1] > threshold).astype(int)

        f1 = self.f1_metric.compute(predictions=predictions, references=labels, average='micro')
        precision = self.precision_metric.compute(predictions=predictions, references=labels)
        recall = self.recall_metric.compute(predictions=predictions, references=labels)

        return {
              "f1": f1['f1'],
              "precision": precision['precision'],
              "recall": recall['recall']
          }

    def train(self, dataset, batch_size=32, epochs=10, flag=False):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            #warmup_steps=300,
            #weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=400,
            #learning_rate=2e-5,            
            #lr_scheduler_type='linear',
            #load_best_model_at_end=True,
            #metric_for_best_model="f1",
        )
        
        optimizer = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            optimizers=(optimizer, lr_scheduler),
            eval_dataset=dataset['test'],
            compute_metrics=self.compute_metrics,
            #callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]
        )

        return self.trainer.train()
    