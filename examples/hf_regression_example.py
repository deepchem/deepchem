import deepchem as dc
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import numpy as np


def main():
    MODEL_NAME = "roberta-base"

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    # Load model with explicit regression setup
    config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                           num_labels=1,
                           problem_type="regression")
    model = RobertaForSequenceClassification(config)
    # DeepChem HuggingFace wrapper with checkpointing enabled
    hf_model = dc.models.torch_models.HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="regression",
        model_dir="./checkpoints_regression")

    # Example dataset
    text_list = [
        "This molecule has a molecular weight of 300 and a logP of 2.5.",
        "This molecule has a molecular weight of 500 and a logP of 5.0.",
        "This molecule has a molecular weight of 100 and a logP of 1.0."
    ]

    labels = np.array([0.5, 1.0, 0.1])

    dataset = dc.data.NumpyDataset(X=np.array(text_list), y=labels)

    # Train
    hf_model.fit(dataset, nb_epoch=3, max_checkpoints_to_keep=1)

    # Load checkpoint and predict
    hf_model.load_from_pretrained()

    # Predict
    predictions = hf_model.predict(dataset)
    print("Regression Predictions:", predictions.flatten())


if __name__ == "__main__":
    main()
