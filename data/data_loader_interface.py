class IDataLoader:
    def load_data(self):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError
