import os

import deepchem as dc
import numpy as np
import pytest

try:
    import torch
    from deepchem.models.torch_models.hf_models import HuggingFaceModel
except ModuleNotFoundError:
    pass


@pytest.fixture
def hf_tokenizer(tmpdir):
    filepath = os.path.join(tmpdir, 'smiles.txt')
    with open(filepath, 'w') as f:
        f.write(
            'CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1\nCC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1\n'
        )
        f.write(
            'COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC\nOCCn1cc(CNc2cccc3c2CCCC3)nn1\n'
        )
        f.write(
            'CCCCCCc1ccc(C#Cc2ccc(C#CC3=CC=C(CCC)CC3)c(C3CCCCC3)c2)c(F)c1\nO=C(NCc1ccc(F)cc1)N1CC=C(c2c[nH]c3ccccc23)CC1\n'
        )
    from tokenizers import ByteLevelBPETokenizer
    from transformers.models.roberta import RobertaTokenizerFast
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=filepath,
                    vocab_size=1_000,
                    min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer_path = os.path.join(tmpdir, 'tokenizer')
    os.makedirs(tokenizer_path)
    tokenizer.save_model(tokenizer_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


@pytest.mark.torch
def test_pretraining(hf_tokenizer, smiles_regression_dataset):
    from deepchem.models.torch_models.hf_models import HuggingFaceModel
    from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForMaskedLM(config)

    hf_model = HuggingFaceModel(model=model,
                                tokenizer=hf_tokenizer,
                                task='mlm',
                                device=torch.device('cpu'))
    loss = hf_model.fit(smiles_regression_dataset, nb_epoch=1)

    assert loss


@pytest.mark.torch
def test_hf_model_regression(hf_tokenizer, smiles_regression_dataset):
    from transformers.models.roberta import (RobertaConfig,
                                             RobertaForSequenceClassification)

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size,
                           problem_type='regression',
                           num_labels=1)
    model = RobertaForSequenceClassification(config)
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=hf_tokenizer,
                                task='regression',
                                device=torch.device('cpu'))
    hf_model.fit(smiles_regression_dataset, nb_epoch=1)
    result = hf_model.predict(smiles_regression_dataset)

    assert result.all()
    score = hf_model.evaluate(smiles_regression_dataset,
                              metrics=dc.metrics.Metric(dc.metrics.mae_score))
    assert score


@pytest.mark.torch
def test_hf_model_classification(hf_tokenizer, smiles_regression_dataset):
    y = np.random.choice([0, 1], size=smiles_regression_dataset.y.shape)
    dataset = dc.data.NumpyDataset(X=smiles_regression_dataset.X,
                                   y=y,
                                   w=smiles_regression_dataset.w,
                                   ids=smiles_regression_dataset.ids)

    from transformers import RobertaConfig, RobertaForSequenceClassification

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForSequenceClassification(config)
    hf_model = HuggingFaceModel(model=model,
                                task='classification',
                                tokenizer=hf_tokenizer,
                                device=torch.device('cpu'))

    hf_model.fit(dataset, nb_epoch=1)
    result = hf_model.predict(dataset)
    assert result.all()
    score = hf_model.evaluate(dataset,
                              metrics=dc.metrics.Metric(dc.metrics.f1_score))
    assert score


@pytest.mark.torch
def test_load_from_pretrained(tmpdir, hf_tokenizer):
    # Create pretrained model
    from transformers.models.roberta import (RobertaConfig, RobertaForMaskedLM,
                                             RobertaForSequenceClassification)

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForMaskedLM(config)
    pretrained_model = HuggingFaceModel(model=model,
                                        tokenizer=hf_tokenizer,
                                        task='mlm',
                                        model_dir=tmpdir,
                                        device=torch.device('cpu'))
    pretrained_model.save_checkpoint()

    # Create finetuning model
    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size,
                           problem_type='regression',
                           num_labels=1)
    model = RobertaForSequenceClassification(config)
    finetune_model = HuggingFaceModel(model=model,
                                      tokenizer=hf_tokenizer,
                                      task='regression',
                                      model_dir=tmpdir,
                                      device=torch.device('cpu'))

    # Load pretrained model
    finetune_model.load_from_pretrained()

    # check weights match
    pretrained_model_state_dict = pretrained_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrained_base_model_keys = [
        key for key in pretrained_model_state_dict.keys() if 'roberta' in key
    ]
    matches = [
        torch.allclose(pretrained_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrained_base_model_keys
    ]

    assert all(matches)


@pytest.mark.torch
def test_model_save_reload(tmpdir, hf_tokenizer):
    from transformers.models.roberta import (RobertaConfig,
                                             RobertaForSequenceClassification)

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForSequenceClassification(config)
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=hf_tokenizer,
                                task='classification',
                                model_dir=tmpdir,
                                device=torch.device('cpu'))
    hf_model._ensure_built()
    hf_model.save_checkpoint()

    model = RobertaForSequenceClassification(config)
    hf_model2 = HuggingFaceModel(model=model,
                                 tokenizer=hf_tokenizer,
                                 task='classification',
                                 model_dir=tmpdir,
                                 device=torch.device('cpu'))
    hf_model2.restore()

    old_state = hf_model.model.state_dict()
    new_state = hf_model2.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys should match
    assert all(matches)


@pytest.mark.torch
def test_load_from_hf_checkpoint():
    from transformers.models.t5 import T5Config, T5Model
    config = T5Config()
    model = T5Model(config)
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=None,
                                task=None,
                                device=torch.device('cpu'))
    old_state_dict = hf_model.model.state_dict()
    hf_model_checkpoint = 't5-small'
    hf_model.load_from_pretrained(hf_model_checkpoint, from_hf_checkpoint=True)
    new_state_dict = hf_model.model.state_dict()
    not_matches = [
        not torch.allclose(old_state_dict[key], new_state_dict[key])
        for key in old_state_dict.keys()
    ]

    # keys should not match
    assert all(not_matches)
