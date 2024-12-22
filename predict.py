from inference.inference import Inference


def predict_labels(
    input_sentence,
    model_path="model/pretrained/ner_model.h5",
    mappings_path="model/mappings",
):

    inference = Inference(model_path=model_path, mappings_path=mappings_path)
    predicted_labels = inference.predict(input_sentence)

    for word, label in zip(input_sentence, predicted_labels):
        print(f"{word}: {label}")


if __name__ == "__main__":
    input_sentence = ["Today", "Ayoub", "is", "in", "Morocco"]
    predict_labels(input_sentence)
