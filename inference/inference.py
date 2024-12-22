import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Inference:
    def __init__(self, model_path, mappings_path, max_sequence_length=50):
        self.model_path = model_path
        self.mappings_path = mappings_path
        self.max_sequence_length = max_sequence_length

        self.model = self.load_model()
        self.word_to_idx, self.idx_to_word = self.load_mappings("word_to_idx")
        self.tag_to_idx, self.idx_to_tag = self.load_mappings("tag_to_idx")

    def load_model(self):
        return load_model(self.model_path)

    def load_mappings(self, mapping_type):
        with open(f"{self.mappings_path}/{mapping_type}.json", "r") as f:
            mapping = json.load(f)
        idx_to_mapping = {v: k for k, v in mapping.items()}
        return mapping, idx_to_mapping

    def preprocess_input(self, sentence):
        sentence_indices = [
            self.word_to_idx.get(word, self.word_to_idx["ENDPAD"]) for word in sentence
        ]

        return pad_sequences(
            [sentence_indices],
            maxlen=self.max_sequence_length,
            padding="post",
            value=self.word_to_idx["ENDPAD"],
        )

    def predict(self, sentence):
        input_sequence = self.preprocess_input(sentence)

        pred = self.model.predict(input_sequence)

        pred_tags = np.argmax(pred, axis=-1)[
            0
        ]  # get the predicted tag IDs for the sentence
        predicted_labels = [
            self.idx_to_tag.get(tag_id, "O") for tag_id in pred_tags
        ]  # 'O' for unknown tags

        return predicted_labels
