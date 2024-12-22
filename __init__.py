from training.trainer import NERModelTrainer


if __name__ == "__main__":
    file_path = "/content/drive/MyDrive/NER/ner_dataset.csv"
    save_path = "/content/drive/MyDrive"

    ner_trainer = NERModelTrainer(file_path)
    ner_trainer.train_model(epochs=5)
    ner_trainer.model.save(save_path)

    test_results = ner_trainer.evaluate_model()
    print("Test Results: ", test_results)
