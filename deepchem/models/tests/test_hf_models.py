import os
import pytest
import deepchem as dc
import pandas as pd


@pytest.mark.torch
def test_bucket_generator():
    "Tests bucket_generator method of HuggingFaceModel class"

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir,
                                "../../models/tests/assets/zinc100.csv")
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
    dataset_path = os.path.join(current_dir,
                                "../../models/tests/assets/zinc100.csv")
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
    input_file = os.path.join(
        cwd, '../../models/tests/assets/multitask_regression.csv')

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
    dataset_path = os.path.join(current_dir,
                                "../../models/tests/assets/zinc100.csv")
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
    dataset_path = os.path.join(current_dir,
                                "../../models/tests/assets/zinc100.csv")
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
