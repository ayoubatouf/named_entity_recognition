from model.utils import save_mappings, save_model
from training.trainer import NERModelTrainer


def train_and_evaluate_ner_model(
    data_path="raw/ner_dataset.csv",
    model_path="model/pretrained",
    mappings_path="model/mappings",
    epochs=5,
):

    ner_trainer = NERModelTrainer(data_path)
    ner_trainer.train_model(epochs=epochs)

    save_mappings(ner_trainer.word_to_idx, ner_trainer.tag_to_idx, mappings_path)
    save_model(ner_trainer.model, model_path)

    test_results = ner_trainer.evaluate_model()
    print("Test Results: ", test_results)

    return test_results


if __name__ == "__main__":
    train_and_evaluate_ner_model()
