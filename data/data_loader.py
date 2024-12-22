import pandas as pd
from data.data_loader_interface import IDataLoader


class DatasetLoader(IDataLoader):
    def __init__(self, file_path, max_sequence_length=50):
        self.file_path = file_path
        self.max_sequence_length = max_sequence_length
        self.data = None
        self.sentences = None
        self.words = None
        self.tags = None
        self.num_words = 0
        self.num_tags = 0

    def load_data(self):
        self.data = pd.read_csv(self.file_path, encoding="ISO-8859-1")
        self.data = self.data.dropna(subset=["Word"]).fillna(method="ffill")

    def preprocess_data(self):
        self.words = list(set(self.data["Word"].values)) + ["ENDPAD"]
        self.tags = list(set(self.data["Tag"].values))
        self.sentences = self.extract_sentences()

        self.num_words = len(self.words)
        self.num_tags = len(self.tags)

    def extract_sentences(self):
        aggregate_function = lambda group: [
            (word, pos, tag)
            for word, pos, tag in zip(
                group["Word"].tolist(), group["POS"].tolist(), group["Tag"].tolist()
            )
        ]
        grouped_data = self.data.groupby("Sentence #").apply(aggregate_function)
        return [sentence for sentence in grouped_data]

    def get_word_to_index_mapping(self):
        return {word: idx + 1 for idx, word in enumerate(self.words)}

    def get_tag_to_index_mapping(self):
        return {tag: idx for idx, tag in enumerate(self.tags)}
