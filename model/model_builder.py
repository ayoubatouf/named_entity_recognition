from model.model_builder_interface import IModelBuilder
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GRU,
    Dense,
    SpatialDropout1D,
    TimeDistributed,
)
from tensorflow.keras.models import Model


class GRUModelBuilder(IModelBuilder):
    def __init__(self, num_words, num_tags, max_sequence_length):
        self.num_words = num_words
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length

    def build_model(self):
        input_layer = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(
            input_dim=self.num_words,
            output_dim=self.max_sequence_length,
            input_length=self.max_sequence_length,
        )(input_layer)
        dropout_layer = SpatialDropout1D(0.1)(embedding_layer)
        gru_layer = GRU(units=100, return_sequences=True, recurrent_dropout=0.1)(
            dropout_layer
        )
        output_layer = TimeDistributed(Dense(self.num_tags, activation="softmax"))(
            gru_layer
        )
        model = Model(input_layer, output_layer)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model
