import deepchem as dc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np


def main():
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               num_labels=1)
    hf_model = dc.models.torch_models.HuggingFaceModel(model=model,
                                                       tokenizer=tokenizer,
                                                       task="regression")
    text_list = [
        "This molecule has a molecular weight of 300 and a logP of 2.5.",
        "This molecule has a molecular weight of 500 and a logP of 5.0.",
        "This molecule has a molecular weight of 100 and a logP of 1.0."
    ]
    labels = np.array([0.5, 1.0, 0.1])  # Example regression targets
    dataset = dc.data.NumpyDataset(text_list, labels)
    # Fine-tune the HuggingFace model using DeepChem
    hf_model.fit(dataset, nb_epoch=3)
    # Predict regression values for the input texts
    predictions = hf_model.predict(dataset)
    print("Regression Predictions:", predictions.flatten())


if __name__ == "__main__":
    main()
