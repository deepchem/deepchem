import deepchem as dc
import pytest
import numpy as np


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
    import torch

    generator = LSTMGenerator()

    gen_iter = generator.default_generator(["CCC"])
    value = list(gen_iter)[0]
    assert torch.equal(value[0], torch.Tensor([[101, 21362, 1658]]))
    assert torch.equal(value[1], torch.Tensor([[21362, 1658, 102]]))

@pytest.mark.torch
def test_fit_model():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator
    import torch

    generator = LSTMGenerator(model_dir="./assets/lstm_generator")

    loss1, loss2 = generator.fit(["CCC"], checkpoint_interval=1, max_checkpoints_to_keep=1)
    assert type(loss1) == float 
    assert type(loss2) == float 

@pytest.mark.torch
def test_load_pretrained():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator
    import torch

    generator = LSTMGenerator()

    generator.load_from_pretrained("./assets/lstm_generator")
    assert generator._built

@pytest.mark.torch
def test_sampling():
    from deepchem.models.torch_models.lstm_generator_models import LSTMGenerator
    import torch

    generator = LSTMGenerator()

    generator.load_from_pretrained("./assets/lstm_generator")
    random_gens = generator.sample(3, max_len=10)
    assert len(random_gens) == 3 

