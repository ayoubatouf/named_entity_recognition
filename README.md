#  Named entity recognition

## Overview

This project implements a neural network-based model for predicting `tags for words` in sentences, leveraging a `Gated Recurrent Unit (GRU)` architecture. Designed for tasks such as named entity recognition, the model efficiently captures contextual relationships within sentences. With robust preprocessing and training, it achieves 98% accuracy on the test set. The dataset used : `https://www.kaggle.com/datasets/namanj27/ner-dataset`

## Tags used

| **Tag**   | **Meaning**                  |
|-----------|------------------------------|
| B-org     | Beginning of an organization |
| B-gpe     | Beginning of a geopolitical entity |
| B-eve     | Beginning of an event        |
| O         | Outside (no specific tag)    |
| B-geo     | Beginning of a geographic location |
| I-org     | Inside an organization       |
| I-nat     | Inside a natural object      |
| B-tim     | Beginning of a time expression |
| I-per     | Inside a person’s name       |
| I-tim     | Inside a time expression     |
| B-per     | Beginning of a person’s name |
| B-nat     | Beginning of a natural object |
| B-art     | Beginning of an artifact     |
| I-eve     | Inside an event              |
| I-geo     | Inside a geographic location |
| I-gpe     | Inside a geopolitical entity |
| I-art     | Inside an artifact           |


## Usage
To train the model, run `train.py` in the root directory. This will generate a pre-trained model in H5 format and save the mappings of words and tags to indexes in JSON format. To perform inference on a new sentence, use `predict.py`.

```
input_sentence = ["Today", "Ayoub", "is", "in", "Morocco"]


1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step

Today: B-tim

Ayoub: B-per

is: O

in: O

Morocco: B-geo
```
