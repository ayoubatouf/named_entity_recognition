import os
import json


def save_mappings(word_to_idx, tag_to_idx, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "/word_to_idx.json", "w") as f:
        json.dump(word_to_idx, f)

    with open(save_path + "/tag_to_idx.json", "w") as f:
        json.dump(tag_to_idx, f)


def save_model(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.save(save_path + "/ner_model.h5")
