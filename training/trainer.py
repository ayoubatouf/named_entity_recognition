import numpy as np
from sklearn.model_selection import train_test_split
from data.data_loader import DatasetLoader
from model.model_builder import GRUModelBuilder
from preprocessing.sentence_processor import SentenceProcessor
from training.callback.callback_manager import CallbackManager
from training.trainer_interface import INERModelTrainer


class NERModelTrainer(INERModelTrainer):
    def __init__(self, file_path, max_sequence_length=50, test_size=0.1):
        self.dataset_loader = DatasetLoader(file_path, max_sequence_length)
        self.dataset_loader.load_data()
        self.dataset_loader.preprocess_data()

        self.word_to_idx = self.dataset_loader.get_word_to_index_mapping()
        self.tag_to_idx = self.dataset_loader.get_tag_to_index_mapping()
        sentence_processor = SentenceProcessor(
            self.dataset_loader.sentences,
            self.word_to_idx,
            self.tag_to_idx,
            max_sequence_length,
        )
        self.X = sentence_processor.process_sentences()
        self.y = sentence_processor.process_tags()

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=1
        )

        self.model_builder = GRUModelBuilder(
            self.dataset_loader.num_words,
            self.dataset_loader.num_tags,
            max_sequence_length,
        )
        self.model = self.model_builder.build_model()

        self.callback_manager = CallbackManager()

    def train_model(self, epochs=5, batch_size=64):
        history = self.model.fit(
            self.x_train,
            np.array(self.y_train),
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=self.callback_manager.get_callbacks(),
        )
        return history

    def evaluate_model(self):
        return self.model.evaluate(self.x_test, np.array(self.y_test))
