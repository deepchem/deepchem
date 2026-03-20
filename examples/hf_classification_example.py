import deepchem as dc
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import numpy as np


def main():
    MODEL_NAME = "roberta-base"

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    # Load model with explicit classification setup
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        num_labels=2,
    )

    model = RobertaForSequenceClassification(config)

    # DeepChem HuggingFace wrapper with checkpointing enabled
    hf_model = dc.models.torch_models.HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        task="classification",
        model_dir="./checkpoints_classification")

    # Example dataset
    text_list = [
        "This molecule is very toxic.", "This molecule is not toxic at all."
    ]

    labels = np.array([1, 0])

    dataset = dc.data.NumpyDataset(X=np.array(text_list), y=labels)

    # Train
    hf_model.fit(dataset, nb_epoch=3)

    # Load model from checkpoint and predict
    hf_model.load_from_pretrained()

    predictions = hf_model.predict(dataset)
    predictions = np.argmax(predictions, axis=1)

    print("Classification Prediction Classes:", predictions)

    metric = dc.metrics.Metric(dc.metrics.f1_score)
    print("Evaluation:", hf_model.evaluate(dataset, [metric]))


if __name__ == "__main__":
    main()
