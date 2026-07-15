import os
import deepchem as dc
import numpy as np
import pytest
from deepchem.models.torch_models.tests.conftest_dnabert import (  # noqa: F401
    genomic_regression_dataset, genomic_multitask_regression_dataset)

try:
    import torch
    from deepchem.models.torch_models.dnabert import Dnabert
except ModuleNotFoundError:
    pass


@pytest.mark.hf
def test_dnabert_pretraining(genomic_regression_dataset,
                             genomic_multitask_regression_dataset):
    # Pretraining in MLM mode
    from deepchem.models.torch_models.dnabert import Dnabert

    model = Dnabert(task='mlm')
    loss = model.fit(genomic_regression_dataset, nb_epoch=1)
    assert loss

    # Pretraining in Multitask Regression Mode
    model = Dnabert(task='mtr', n_tasks=2)
    loss = model.fit(genomic_multitask_regression_dataset, nb_epoch=1)
    assert loss


@pytest.mark.hf
def test_dnabert_finetuning(genomic_regression_dataset,
                            genomic_multitask_regression_dataset):
    # test regression
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    model = Dnabert(task='regression', tokenizer_path=tokenizer_path)
    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)
    loss = model.fit(genomic_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(genomic_regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(genomic_regression_dataset)
    assert prediction.shape == genomic_regression_dataset.y.shape

    # test multitask regression
    model = Dnabert(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)
    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)
    loss = model.fit(genomic_multitask_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(genomic_multitask_regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(genomic_multitask_regression_dataset)
    assert prediction.shape == genomic_multitask_regression_dataset.y.shape

    # test classification
    y = np.random.choice([0, 1], size=genomic_regression_dataset.y.shape)
    dataset = dc.data.NumpyDataset(X=genomic_regression_dataset.X,
                                   y=y,
                                   w=genomic_regression_dataset.w,
                                   ids=genomic_regression_dataset.ids)
    model = Dnabert(task='classification', tokenizer_path=tokenizer_path)
    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    # logit scores
    assert prediction.shape == (dataset.y.shape[0], 2)


@pytest.mark.hf
def test_dnabert_load_from_pretrained(tmpdir, genomic_regression_dataset):
    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    pretrain_model = Dnabert(task='mlm',
                             tokenizer_path=tokenizer_path,
                             model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()

    finetune_model = Dnabert(task='regression',
                             tokenizer_path=tokenizer_path,
                             model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)


@pytest.mark.hf
def test_dnaber_save_reload(tmpdir):
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    model = Dnabert(task='regression',
                    tokenizer_path=tokenizer_path,
                    model_dir=tmpdir)
    model._ensure_built()
    model.save_checkpoint()

    model_new = Dnabert(task='regression',
                        tokenizer_path=tokenizer_path,
                        model_dir=tmpdir)
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)


@pytest.mark.hf
def test_dnabert_load_weights_from_hf_hub():
    pretrained_model_path = 'IronHead44/DNABERT-2-117M'
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    model = Dnabert(task='regression', tokenizer_path=tokenizer_path)
    old_model_id = id(model.model)
    model.load_from_pretrained(pretrained_model_path, from_hf_checkpoint=True)
    new_model_id = id(model.model)
    # new model's model attribute is an entirely new model initiated by AutoModel.load_from_pretrained
    # and hence it should have a different identifier
    assert old_model_id != new_model_id


@pytest.mark.hf
def test_dnabert_finetuning_multitask_classification():
    # test multitask classification with 10 tasks
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    sequences = [
        "ATGCGTACGTTAGCTAGCATGCGTACG",
        "GGCTAACCGTATCGGATCAAGTCCTAG",
        "TTAAGCCGTACGATCGATCGATCGATCG",
        "CCGATCGATCGATCGATCGATCGATCGA",
        "ATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "AATTCCGGAATTCCGGAATTCCGGAATT",
        "CCGGTTAACCGGTTAACCGGTTAACCGG",
        "TTAGGCCAATTAGGCCAATTAGGCCAAT",
        "GGCCAATTGGCCAATTGGCCAATTGGCC",
    ]
    # 10 binary classification tasks, 10 samples
    np.random.seed(42)
    y = np.random.choice([0, 1], size=(10, 10))
    dataset = dc.data.NumpyDataset(X=np.array(sequences), y=y)

    model = Dnabert(task='classification',
                    tokenizer_path=tokenizer_path,
                    n_tasks=10)
    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    assert prediction.shape == (dataset.y.shape[0], 10)


@pytest.mark.hf
def test_dnabert_finetuning_multitask_regression():
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    sequences = [
        "ATGCGTACGTTAGCTAGCATGCGTACG",
        "GGCTAACCGTATCGGATCAAGTCCTAG",
        "TTAAGCCGTACGATCGATCGATCGATCG",
        "CCGATCGATCGATCGATCGATCGATCGA",
        "ATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "AATTCCGGAATTCCGGAATTCCGGAATT",
        "CCGGTTAACCGGTTAACCGGTTAACCGG",
        "TTAGGCCAATTAGGCCAATTAGGCCAAT",
        "GGCCAATTGGCCAATTGGCCAATTGGCC",
    ]
    # 10 regression tasks, 10 samples
    np.random.seed(42)
    y = np.random.randn(10, 10)
    dataset = dc.data.NumpyDataset(X=np.array(sequences), y=y)

    model = Dnabert(task='regression',
                    tokenizer_path=tokenizer_path,
                    n_tasks=10)
    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_squared_error))
    assert eval_score, loss
    prediction = model.predict(dataset)
    assert prediction.shape == dataset.y.shape
