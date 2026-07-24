import os
import numpy as np
import pandas as pd
import pytest

import deepchem as dc

try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
except ImportError:
    gpu_available = False

try:
    from deepchem.models.lightning import LightningTorchModel
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True

pytestmark = [
    pytest.mark.skipif(not gpu_available,
                       reason="Multi-GPU testing requires at least 2 GPUs"),
    pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                       reason="PyTorch Lightning is not installed")
]


@pytest.fixture(scope="function")
def smiles_regression_dataset(tmpdir):
    """Creates a single-task regression dataset with two SMILES molecules and continuous labels."""
    smiles = [
        "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
        "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
    ]
    labels = [3.112, 2.432]
    df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    filepath = os.path.join(tmpdir, 'smiles_reg.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset


@pytest.mark.hf
def test_olmo_lightning_fit_and_predict(smiles_regression_dataset):
    """Test QLoRA regression training and prediction via PyTorch Lightning DDP."""
    from deepchem.models.torch_models.olmo import Olmo

    tokenizer_path = 'allenai/OLMo-1B-hf'

    model = Olmo(task_type="regression",
                 tokenizer_path=tokenizer_path,
                 finetune_strategy='qlora',
                 torch_dtype=torch.float16,
                 batch_size=2)

    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)

    dataset = smiles_regression_dataset

    trainer = LightningTorchModel(model=model,
                                  batch_size=2,
                                  max_epochs=1,
                                  enable_progress_bar=True,
                                  accelerator="gpu",
                                  strategy="ddp",
                                  devices=-1,
                                  log_every_n_steps=1)

    trainer.fit(dataset, num_workers=0)
    predictions = trainer.predict(dataset)

    assert len(predictions) > 0
    assert isinstance(predictions, np.ndarray)
    # The final prediction shape should be (n_samples, n_tasks)
    assert predictions.shape == (2, 1)
