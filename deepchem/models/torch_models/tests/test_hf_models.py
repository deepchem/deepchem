import os
import deepchem as dc
import numpy as np
import pytest
import pandas as pd

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


@pytest.mark.hf
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


@pytest.mark.hf
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


@pytest.mark.hf
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


@pytest.mark.hf
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


@pytest.mark.hf
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


@pytest.mark.hf
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


@pytest.mark.hf
def test_fill_mask_IO(tmpdir, hf_tokenizer):
    from transformers import (RobertaConfig, RobertaForMaskedLM)

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForMaskedLM(config)
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=hf_tokenizer,
                                task='mlm',
                                model_dir=tmpdir,
                                device=torch.device('cpu'))
    hf_model._ensure_built()

    test_string = "CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1"
    tokenized_test_string = hf_tokenizer(test_string)
    tokenized_test_string.input_ids[1] = hf_tokenizer.mask_token_id
    masked_test_string = hf_tokenizer.decode(tokenized_test_string.input_ids)

    results = hf_model.fill_mask([masked_test_string, masked_test_string])

    # Test 1. Ensure that the correct number of filled items comes out.
    assert len(results) == 2

    # Test 2. Ensure the types are the expected types
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    assert isinstance(results[0][0], dict)


@pytest.mark.hf
def test_fill_mask_fidelity(tmpdir, hf_tokenizer):
    from transformers import (RobertaConfig, RobertaForMaskedLM)

    config = RobertaConfig(vocab_size=hf_tokenizer.vocab_size)
    model = RobertaForMaskedLM(config)
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=hf_tokenizer,
                                task='mlm',
                                model_dir=tmpdir,
                                device=torch.device('cpu'))
    hf_model._ensure_built()

    test_string = "CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1"
    tokenized_test_string = hf_tokenizer(test_string)
    tokenized_test_string.input_ids[1] = hf_tokenizer.mask_token_id
    masked_test_string = hf_tokenizer.decode(tokenized_test_string.input_ids)

    results = hf_model.fill_mask([masked_test_string, masked_test_string])

    for result in results:
        for filled in result:
            # Test 1. Check that the right keys exist
            assert 'sequence' in filled
            assert 'score' in filled
            assert 'token' in filled
            assert 'token_str' in filled

            # Test 2. Check that the scores are probabilities
            assert filled['score'] < 1
            assert filled['score'] >= 0

            # Test 3. Check that the infilling went to the right spot
            assert filled['sequence'].startswith(f'<s>{filled["token_str"]}')


@pytest.mark.hf
def test_load_from_pretrained_with_diff_task(tmpdir):
    # Tests loading a pretrained model where the weight shape in last layer
    # (the final projection layer) of the pretrained model does not match
    # with the weight shape in new model.
    from deepchem.models.torch_models import Chemberta
    model = Chemberta(task='mtr', n_tasks=10, model_dir=tmpdir)
    model.save_checkpoint()

    model = Chemberta(task='regression', n_tasks=20)
    model.load_from_pretrained(model_dir=tmpdir)


@pytest.mark.torch
def test_bucket_generator():
    "Tests bucket_generator method of HuggingFaceModel class"

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, "../../tests/assets/zinc100.csv")
    data = pd.read_csv(dataset_path)
    df = pd.DataFrame(data)
    df['smiles_len'] = data['smiles'].apply(lambda x: len(x))
    mode = df['smiles_len'].mode()[0]
    avg = df['smiles_len'].mean()
    max = df['smiles_len'].max()
    min = df['smiles_len'].min()
    tail = 49
    loader = dc.data.CSVLoader([],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(dataset_path)
    from deepchem.models.torch_models.hf_models import HuggingFaceModel
    from transformers.models.roberta import RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "seyonec/PubChem10M_SMILES_BPE_60k")
    config = RobertaConfig(tokenizer.vocab_size)
    model = RobertaForMaskedLM(config)
    hf_model = HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                task='mlm',
                                model_dir='model-dir',
                                batch_size=10)
    bucket_gen = hf_model.bucket_generator(
        dataset=dataset,
        mode_data_length=mode,
        avg_data_length=avg,
        max_data_length=max,
        min_data_length=min,
        tail_length=tail,
        b0_max=5,
        b1_max=2,
        b2_max=2,
        b3_max=4,
    )

    sum = 0
    bucket = next(bucket_gen)
    for i in bucket:
        sum += len(i)
    assert len(bucket[0]) == 3
    assert len(bucket[1]) == 1
    assert len(bucket[2]) == 0
    assert len(bucket[3]) == 0
    while True:
        try:
            bucket = next(bucket_gen)
            for i in bucket:
                sum += len(i)
        except StopIteration:
            break
    assert sum == 100


@pytest.mark.torch
def test_bucket_fit_generator():
    "Tests bucket_fit_generator method of HuggingFaceModel class"

    import torch
    seed = 34
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Pretraining in MLM mode
    from deepchem.models.torch_models.chemberta import Chemberta

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, "../../tests/assets/zinc100.csv")
    loader = dc.data.CSVLoader([],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(dataset_path)
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    model = Chemberta(task='mlm', tokenizer_path=tokenizer_path)
    data = pd.read_csv(dataset_path)
    df = pd.DataFrame(data)
    df['smiles_len'] = data['smiles'].apply(lambda x: len(x))
    mode = df['smiles_len'].mode()[0]
    avg = df['smiles_len'].mean()
    max = df['smiles_len'].max()
    min = df['smiles_len'].min()
    tail = 49
    loss = model.fit(dataset,
                     nb_epoch=1,
                     enable_bucketing=True,
                     mode_data_length=mode,
                     avg_data_length=avg,
                     max_data_length=max,
                     min_data_length=min,
                     tail_length=tail,
                     b0_max=5,
                     b1_max=2,
                     b2_max=2,
                     b3_max=4)
    assert loss

    cwd = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(cwd,
                              '../../tests/assets/multitask_regression.csv')

    loader = dc.data.CSVLoader(tasks=['task0', 'task1'],
                               feature_field='smiles',
                               featurizer=dc.feat.DummyFeaturizer())
    mtr_dataset = loader.create_dataset(input_file)

    # Pretraining in Multitask Regression Mode
    model = Chemberta(task='mtr', tokenizer_path=tokenizer_path, n_tasks=2)
    loss = model.fit(mtr_dataset,
                     nb_epoch=1,
                     enable_bucketing=True,
                     mode_data_length=mode,
                     avg_data_length=avg,
                     max_data_length=max,
                     min_data_length=min,
                     tail_length=tail,
                     b0_max=5,
                     b1_max=2,
                     b2_max=2,
                     b3_max=4)
    assert loss


def test_fit_generator_without_bucketing():
    "Tests fit_generator method of HuggingFaceModel class without bucketing."

    # Pretraining in MLM mode
    from deepchem.models.torch_models.chemberta import Chemberta

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, "../../tests/assets/zinc100.csv")
    loader = dc.data.CSVLoader([],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(dataset_path)
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    model = Chemberta(task='mlm', tokenizer_path=tokenizer_path)
    loss = model.fit(dataset, nb_epoch=1)
    assert loss


def test_assertion_data_lengths():
    "Tests if the assertion errors are showing up in case of wrong data length inputs"

    # Pretraining in MLM mode
    from deepchem.models.torch_models.chemberta import Chemberta

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, "../../tests/assets/zinc100.csv")
    loader = dc.data.CSVLoader([],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(dataset_path)
    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    model = Chemberta(task='mlm', tokenizer_path=tokenizer_path)

    mode = 50
    avg = 12
    max = 60
    min = 10
    tail = 13
    try:
        model.fit(dataset,
                  nb_epoch=1,
                  enable_bucketing=True,
                  mode_data_length=mode,
                  avg_data_length=avg,
                  max_data_length=max,
                  min_data_length=min,
                  tail_length=tail,
                  b0_max=5,
                  b1_max=2,
                  b2_max=2,
                  b3_max=4)
    except AssertionError as e:
        assert e.__str__(
        ), "tail length should be >= mode or average data length"

    mode = 12
    avg = 50
    max = 60
    min = 10
    tail = 13
    try:
        model.fit(dataset,
                  nb_epoch=1,
                  enable_bucketing=True,
                  mode_data_length=mode,
                  avg_data_length=avg,
                  max_data_length=max,
                  min_data_length=min,
                  tail_length=tail,
                  b0_max=5,
                  b1_max=2,
                  b2_max=2,
                  b3_max=4)
    except AssertionError as e:
        assert e.__str__(
        ), "tail length should be >= mode or average data length"

    mode = 12
    avg = 30
    max = 60
    min = 100
    tail = 50
    try:
        model.fit(dataset,
                  nb_epoch=1,
                  enable_bucketing=True,
                  mode_data_length=mode,
                  avg_data_length=avg,
                  max_data_length=max,
                  min_data_length=min,
                  tail_length=tail,
                  b0_max=5,
                  b1_max=2,
                  b2_max=2,
                  b3_max=4)
    except AssertionError as e:
        assert e.__str__(), "min length should be minimum of all parameters"

    mode = 12
    avg = 30
    max = 60
    min = 10
    tail = 500
    try:
        model.fit(dataset,
                  nb_epoch=1,
                  enable_bucketing=True,
                  mode_data_length=mode,
                  avg_data_length=avg,
                  max_data_length=max,
                  min_data_length=min,
                  tail_length=tail,
                  b0_max=5,
                  b1_max=2,
                  b2_max=2,
                  b3_max=4)
    except AssertionError as e:
        assert e.__str__(), "max length should be maximum of all parameters"
