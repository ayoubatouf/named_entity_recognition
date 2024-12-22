class INERModelTrainer:
    def train_model(self, epochs=5, batch_size=64):
        raise NotImplementedError

    def evaluate_model(self):
        raise NotImplementedError
