import os
import pandas as pd
import pytest

import deepchem as dc

SMILES = [
    "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
]


@pytest.fixture(scope="function")
def smiles_regression_dataset(tmpdir):
    """Creates a single-task regression dataset with two SMILES molecules and continuous labels."""
    labels = [3.112, 2.432]
    df = pd.DataFrame(list(zip(SMILES, labels)), columns=["smiles", "task1"])
    filepath = os.path.join(tmpdir, "smiles_reg.csv")
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    return loader.create_dataset(filepath)


@pytest.mark.hf
@pytest.mark.parametrize("strategy", ["lora", "qlora"])
def test_olmo_lora_qlora_applied_at_init(strategy):
    """Test that LoRA/QLoRA adapters are applied at init with the correct trainable parameter structure."""
    from peft import PeftModel
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(task_type="regression",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=1,
                 finetune_strategy=strategy,
                 torch_dtype="float16")

    assert isinstance(model.model, PeftModel)

    trainable = [
        n for n, p in model.model.named_parameters() if p.requires_grad
    ]
    assert len(trainable) > 0
    # All trainable params must be either LoRA adapter weights or the regression head
    assert all("lora_" in n or "score" in n for n in trainable)
    # At least some LoRA adapter params must be trainable
    assert any("lora_" in n for n in trainable)

    # Base transformer backbone weights must be frozen
    frozen = [
        n for n, p in model.model.named_parameters() if not p.requires_grad
    ]
    assert len(frozen) > 0


@pytest.mark.hf
@pytest.mark.parametrize("strategy", ["lora", "qlora"])
def test_olmo_lora_qlora_fit_predict(smiles_regression_dataset, strategy):
    """Test that a LoRA/QLoRA-wrapped regression model can fit and predict."""
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(task_type="regression",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=1,
                 finetune_strategy=strategy,
                 torch_dtype="float16")

    loss = model.fit(smiles_regression_dataset, nb_epoch=1)
    assert loss is not None

    prediction = model.predict(smiles_regression_dataset)
    assert prediction.shape == smiles_regression_dataset.y.shape
