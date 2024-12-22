from preprocessing.sentence_processor_interface import ISentenceProcessor
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentenceProcessor(ISentenceProcessor):
    def __init__(self, sentences, word_to_idx, tag_to_idx, max_sequence_length):
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_sequence_length = max_sequence_length

    def process_sentences(self):
        sentence_indices = [
            [self.word_to_idx[word[0]] for word in sentence]
            for sentence in self.sentences
        ]
        return pad_sequences(
            sentence_indices,
            maxlen=self.max_sequence_length,
            padding="post",
            value=self.word_to_idx["ENDPAD"],
        )

    def process_tags(self):
        tag_indices = [
            [self.tag_to_idx[word[2]] for word in sentence]
            for sentence in self.sentences
        ]
        tag_indices_padded = pad_sequences(
            tag_indices,
            maxlen=self.max_sequence_length,
            padding="post",
            value=self.tag_to_idx["O"],
        )
        return [
            to_categorical(tags, num_classes=len(self.tag_to_idx))
            for tags in tag_indices_padded
        ]
