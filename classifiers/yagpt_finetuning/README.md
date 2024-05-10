# Yandex GPT Fine-Tuning Data

This folder contains datasets and scripts for fine-tuning the Yandex GPT to classify texts into toxic and non-toxic categories.

## Folder Contents

- `train_toxigen.csv`
  - This file is the training dataset used to fine-tune Yandex GPT on the task of toxic texts classification.

- `test_toxigen.csv`
  - This file is the testing dataset used to evaluate the performance of the Yandex GPT after fine-tuning on toxic texts classification.

- `train_toxigen_gpt.jsonl`
  - This is the JSONL version of the training dataset, formatted specifically for compatibility with Yandex GPT's fine-tuning requirements.

- `test_toxigen_gpt.jsonl`
  - This is the JSONL version of the testing dataset, also formatted for Yandex GPT's fine-tuning process.

- `get_jsonl.py`
  - A Python script designed to convert CSV files into the JSONL format required by Yandex GPT for fine-tuning datasets. This script automates the preparation of data for model training.

### Data Format and Labeling

In the CSV files, the `label` column should contain a `1` if the text is toxic and a `0` if it is not. 

## Script Usage

The `get_jsonl.py` script is used to convert CSV files into JSONL format suitable for Yandex GPT's fine-tuning process. To use this script, specify the input CSV file and the desired output JSONL file:

```
python get_jsonl.py --input_file=<path_to_your_csv> --output_file=<desired_path_to_output_jsonl>
```

