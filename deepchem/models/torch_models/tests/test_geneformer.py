import os
import pytest
import numpy as np
import deepchem as dc

try:
    import torch
    from deepchem.models.torch_models.geneformer import Geneformer
except ImportError:
    pass


@pytest.mark.hf
def test_geneformer_finetuning():
    """Test Geneformer fine-tuning on classification and regression tasks."""

    # 1. Create Dummy Data (Pre-featurized token IDs)
    # Vocab size for tiny model = 100. Shape = (10 samples, 5 genes)
    X = np.random.randint(0, 100, size=(10, 5))
    y_class = np.random.randint(0, 2, size=(10,))
    y_reg = np.random.rand(10,)

    classification_dataset = dc.data.NumpyDataset(X, y_class)
    regression_dataset = dc.data.NumpyDataset(X, y_reg)

    # Tiny Config to avoid large downloads
    tiny_config = {
        "vocab_size": 100,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 128
    }

    # Test Classification
    model = Geneformer(task='classification', n_tasks=1, config=tiny_config)
    loss = model.fit(classification_dataset, nb_epoch=1)
    eval_score = model.evaluate(classification_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.accuracy_score))

    assert loss
    prediction = model.predict(classification_dataset)
    # Output shape for binary classification is (N, 2) logits
    assert prediction.shape == (10, 2)

    # Test Regression
    model = Geneformer(task='regression', n_tasks=1, config=tiny_config)
    loss = model.fit(regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))

    assert loss
    prediction = model.predict(regression_dataset)
    assert prediction.shape == (10, 1)


@pytest.mark.hf
def test_geneformer_multitask():
    """Test Geneformer on multitask regression (MTR) and classification."""

    X = np.random.randint(0, 100, size=(10, 5))
    # 2 tasks
    y_mtr = np.random.rand(10, 2)
    y_multiclass = np.random.randint(0, 2, size=(10, 2))

    mtr_dataset = dc.data.NumpyDataset(X, y_mtr)
    multiclass_dataset = dc.data.NumpyDataset(X, y_multiclass)

    tiny_config = {
        "vocab_size": 100,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4
    }

    # Test Multitask Regression
    model = Geneformer(task='regression', n_tasks=2, config=tiny_config)
    loss = model.fit(mtr_dataset, nb_epoch=1)
    prediction = model.predict(mtr_dataset)

    assert loss
    assert prediction.shape == (10, 2)

    # Test Multitask Classification
    model = Geneformer(task='classification', n_tasks=2, config=tiny_config)
    loss = model.fit(multiclass_dataset, nb_epoch=1)
    prediction = model.predict(multiclass_dataset)

    assert loss
    # Output is (N, n_tasks) logits for multi-label
    assert prediction.shape == (10, 2)


@pytest.mark.hf
def test_geneformer_load_from_pretrained(tmpdir):
    """Test loading weights from a DeepChem checkpoint."""

    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')

    tiny_config = {
        "vocab_size": 100,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2
    }

    # 1. Train "Pretrained" Model
    pretrain_model = Geneformer(task='regression',
                                config=tiny_config,
                                model_dir=pretrain_model_dir)
    # Force build
    pretrain_model._ensure_built()
    pretrain_model.save_checkpoint()

    # 2. Load into "Finetune" Model
    finetune_model = Geneformer(task='regression',
                                config=tiny_config,
                                model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # 3. Check weights match
    pretrain_state = pretrain_model.model.state_dict()
    finetune_state = finetune_model.model.state_dict()

    # Compare BERT layers
    matches = [
        torch.allclose(pretrain_state[key], finetune_state[key])
        for key in pretrain_state.keys()
        if 'bert' in key
    ]
    assert all(matches)


@pytest.mark.hf
def test_geneformer_save_reload(tmpdir):
    """Test saving and restoring the model."""

    tiny_config = {
        "vocab_size": 100,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2
    }

    model = Geneformer(task='regression',
                       config=tiny_config,
                       model_dir=str(tmpdir))
    model._ensure_built()
    model.save_checkpoint()

    model_new = Geneformer(task='regression',
                           config=tiny_config,
                           model_dir=str(tmpdir))
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()

    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]
    assert all(matches)
