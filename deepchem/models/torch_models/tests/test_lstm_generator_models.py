import pytest


@pytest.mark.torch
def test_load_lstmneuralnet():
    from deepchem.models.torch_models.lstm_generator_models import LSTMNeuralNet
    import torch

    model = LSTMNeuralNet(vocab_size=12,
                          embedding_dim=8,
                          hidden_dim=4,
                          num_layers=1)

    x = torch.randint(0, 12, (2, 10))

    _ = model(x)


@pytest.mark.torch
def test_default_generator():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator
    from deepchem.data import NumpyDataset
    import torch

    generator = LSTMGenerator(model_dir="./assets/lstm_generator")
    dataset = NumpyDataset(["CCC"])
    gen_iter = generator.default_generator(dataset)
    value = list(gen_iter)[0]
    assert torch.equal(value[0][0], torch.Tensor([[101, 21362, 1658]]))
    assert torch.equal(value[1][0], torch.Tensor([[21362, 1658, 102]]))


@pytest.mark.torch
def test_fit_model():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator
    from deepchem.data import NumpyDataset

    generator = LSTMGenerator(model_dir="./assets/lstm_generator")
    dataset = NumpyDataset(["CCC"])
    loss = generator.fit(dataset,
                         checkpoint_interval=1,
                         max_checkpoints_to_keep=1)
    assert type(loss) is float


@pytest.mark.torch
def test_load_pretrained():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator

    generator = LSTMGenerator()

    generator.load_from_pretrained(source_model=LSTMGenerator(),
                                   model_dir="./assets/lstm_generator")
    assert generator._built


@pytest.mark.torch
def test_sampling():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator

    generator = LSTMGenerator()

    generator.load_from_pretrained(source_model=LSTMGenerator(),
                                   model_dir="./assets/lstm_generator")
    random_gens = generator.sample(3, max_len=10)
    assert len(random_gens) == 3


@pytest.mark.torch
def test_predefined_sampling():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator

    generator = LSTMGenerator(
        allow_custom_init_texts=True)

    generator.load_from_pretrained(source_model=LSTMGenerator(),
                                   model_dir="./assets/lstm_generator")
    random_gens = generator.sample(3, max_len=10,
                                   init_texts="pre")
    assert len(random_gens) == 3
    for gen in random_gens:
        assert gen.startswith("pre")
