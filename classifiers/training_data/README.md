# Training Data

This folder contains labeled training and test datasets used for training classifiers to detect implicit toxicity.

## Folder Contents

- `train_lifetox.csv`
- `test_lifetox.csv`
  - These files are the LifeTox training and testing datasets, translated into Russian using the Yandex Translate API. [More about the LifeTox dataset on Hugging Face](https://huggingface.co/datasets/mbkim/LifeTox)

- `train_toxigen`
- `test_toxigen`
  - These files are the ToxiGen training and testing datasets, also translated into Russian using the Yandex Translate API. [More about the ToxiGen dataset on Hugging Face](https://huggingface.co/datasets/toxigen/toxigen-data).

- `explicit_data.csv`
This file contains the Russian Toxic Comments dataset from Kaggle. It is intended for use in training models to recognize explicitly toxic comments. [View this dataset on Kaggle](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments).

- `adversarial.csv`
  - This custom dataset includes examples that are particularly challenging for classifiers (false positives and false negatives). It contains trigger words but conveys meanings opposite to what the presence of such words might suggest.

