import deepchem as dc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np


def main():
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               num_labels=2)
    hf_model = dc.models.torch_models.HuggingFaceModel(model=model,
                                                       tokenizer=tokenizer,
                                                       task="classification")
    text_list = [
        "This molecule is very toxic.", "This molecule is not toxic at all."
    ]
    labels = [1, 0]  # 1 for toxic, 0 for non-toxic
    dataset = dc.data.NumpyDataset(text_list, labels)
    # Fine-tune the HuggingFace model using DeepChem
    hf_model.fit(dataset, nb_epoch=3)
    # Predict class probabilities for the input texts
    predictions = hf_model.predict(dataset)
    # Convert predicted probabilities to class labels
    predictions = np.argmax(predictions,
                            axis=1)  # Get the predicted class labels
    print("Classification Prediction Classes:", predictions)


if __name__ == "__main__":
    main()
