import os
import deepchem as dc
import numpy as np
import pytest

try:
    import torch
    from deepchem.models.torch_models.chemberta import Chemberta
except ModuleNotFoundError:
    pass


@pytest.mark.hf
def test_chemberta_pretraining(smiles_regression_dataset,
                               smiles_multitask_regression_dataset):
    # Pretraining in MLM mode
    from deepchem.models.torch_models.chemberta import Chemberta

    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    model = Chemberta(task='mlm', tokenizer_path=tokenizer_path)
    loss = model.fit(smiles_regression_dataset, nb_epoch=1)

    assert loss

    # Pretraining in Multitask Regression Mode
    model = Chemberta(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)
    loss = model.fit(smiles_multitask_regression_dataset, nb_epoch=1)
    assert loss


@pytest.mark.hf
def test_chemberta_finetuning(smiles_regression_dataset,
                              smiles_multitask_regression_dataset):
    # test regression
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    model = Chemberta(task='regression', tokenizer_path=tokenizer_path)
    loss = model.fit(smiles_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(smiles_regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(smiles_regression_dataset)
    assert prediction.shape == smiles_regression_dataset.y.shape

    # test multitask regression
    model = Chemberta(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)
    loss = model.fit(smiles_multitask_regression_dataset, nb_epoch=1)
    eval_score = model.evaluate(smiles_multitask_regression_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))
    assert loss, eval_score
    prediction = model.predict(smiles_multitask_regression_dataset)
    assert prediction.shape == smiles_multitask_regression_dataset.y.shape

    # test classification
    y = np.random.choice([0, 1], size=smiles_regression_dataset.y.shape)
    dataset = dc.data.NumpyDataset(X=smiles_regression_dataset.X,
                                   y=y,
                                   w=smiles_regression_dataset.w,
                                   ids=smiles_regression_dataset.ids)
    model = Chemberta(task='classification', tokenizer_path=tokenizer_path)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    # logit scores
    assert prediction.shape == (dataset.y.shape[0], 2)


@pytest.mark.hf
def test_chemberta_load_from_pretrained(tmpdir, smiles_regression_dataset):
    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    pretrain_model = Chemberta(task='mlm',
                               tokenizer_path=tokenizer_path,
                               model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()

    finetune_model = Chemberta(task='regression',
                               tokenizer_path=tokenizer_path,
                               model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'roberta' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)


@pytest.mark.hf
def test_chemberta_save_reload(tmpdir):
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    model = Chemberta(task='regression',
                      tokenizer_path=tokenizer_path,
                      model_dir=tmpdir)
    model._ensure_built()
    model.save_checkpoint()

    model_new = Chemberta(task='regression',
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
def test_chemberta_load_weights_from_hf_hub():
    pretrained_model_path = 'DeepChem/ChemBERTa-77M-MLM'
    tokenizer_path = 'DeepChem/ChemBERTa-77M-MLM'
    model = Chemberta(task='regression', tokenizer_path=tokenizer_path)
    old_model_id = id(model.model)
    model.load_from_pretrained(pretrained_model_path, from_hf_checkpoint=True)
    new_model_id = id(model.model)
    # new model's model attribute is an entirely new model initiated by AutoModel.load_from_pretrained
    # and hence it should have a different identifier
    assert old_model_id != new_model_id


@pytest.mark.hf
def test_chemberta_finetuning_multitask_classification():
    # test multitask classification
    loader = dc.molnet.load_clintox(featurizer=dc.feat.DummyFeaturizer())
    tasks, dataset, transformers = loader
    train, val, test = dataset

    train_sample = train.select(range(10))
    test_sample = test.select(range(10))
    model = Chemberta(task='classification', n_tasks=len(tasks))
    loss = model.fit(train_sample, nb_epoch=1)
    eval_score = model.evaluate(test_sample,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.roc_auc_score))
    assert eval_score, loss
    prediction = model.predict(test_sample)
    # logit scores
    assert prediction.shape == (test_sample.y.shape[0], len(tasks))


@pytest.mark.hf
def test_chemberta_finetuning_multitask_regression():
    # test multitask regression

    cwd = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(cwd,
                              '../../tests/assets/multitask_regression.csv')

    loader = dc.data.CSVLoader(tasks=['task0', 'task1'],
                               feature_field='smiles',
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(input_file)
    model = Chemberta(task='regression', n_tasks=2)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.mean_absolute_error))

    assert loss, eval_score
    prediction = model.predict(dataset)
    assert prediction.shape == dataset.y.shape
