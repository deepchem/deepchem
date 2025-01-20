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
def test_lstm_trainer_prepare_dataset():
    from deepchem.models.torch_models.lstm_generator_models import LSTMTrainer
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    trainer = LSTMTrainer(tokenizer=tokenizer,
                          embedding_dim=8,
                          hidden_dim=4,
                          num_layers=1)

    # initiating without any input params
    _ = LSTMTrainer()

    samples = ["CCCC", "CCCCCC"]
    tensors = trainer.prepare_dataset(samples)
    assert tensors.shape == (2, 5)

    trainer.train(samples,
                  batch_size=1,
                  num_epochs=1,
                  learning_rate=0.01,
                  device="auto")


@pytest.mark.torch
def test_fail_model_save():
    from deepchem.models.torch_models.lstm_generator_models import LSTMTrainer
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    trainer = LSTMTrainer(tokenizer=tokenizer,
                          embedding_dim=8,
                          hidden_dim=4,
                          num_layers=1)

    with pytest.raises(ValueError):
        trainer.save_model("test.pth")


@pytest.mark.torch
def test_lstm_sampler():
    from deepchem.models.torch_models.lstm_generator_models import LSTMSampler
    sampler = LSTMSampler(embedding_dim=8, hidden_dim=4, num_layers=1)
    sampler.load_model_from_ckpt("./assets/lstm_sampler.pth")

    results = sampler.generate(number_of_seq=2, max_len=5)
    assert len(results) == 2

@pytest.mark.torch
def test_fail_lstm_sampler():
    from deepchem.models.torch_models.lstm_generator_models import LSTMSampler
    sampler = LSTMSampler(embedding_dim=8, hidden_dim=4, num_layers=1)
    with pytest.raises(ValueError):
        sampler.load_model_from_ckpt("./assets/lstm_sampler.pth")
        sampler.load_model_from_ckpt("./assets/something.pth")

    with pytest.raises(ValueError):
        sampler_alt = LSTMSampler(embedding_dim=8, hidden_dim=4, num_layers=1)
        sampler_alt.generate(number_of_seq=2, max_len=5)