import numpy as np
import pytest
import deepchem as dc

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from deepchem.models.torch_models.hf_models import HuggingFaceModel


def test_huggingface_model_text_classification_predict_shape():
    X = np.array(["hello world", "deepchem test", "transformers wrapper"])
    y = np.array([0, 1, 0])
    dataset = dc.data.NumpyDataset(X=X, y=y)

    checkpoint = "sshleifer/tiny-distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = 2
    hf_model = AutoModelForSequenceClassification.from_config(config)
    
    dc_model = HuggingFaceModel(
        model=hf_model,
        tokenizer=tokenizer,
        task="classification",
        batch_size=2,
    )

    preds = dc_model.predict(dataset)
    assert preds.shape[0] == len(X)