import torch
import numpy as np
import deepchem as dc


from deepchem.models.torch_models.olmo_causal_lm import OLMoCausalLM


def test_olmo_causal_lm_forward():
    """Test that OLMoCausalLM forward returns a loss when labels are provided."""

    model = OLMoCausalLM(
        model_name = "sshleifer/tiny-gpt2",
        torch_dtype = torch.float32,
        device_map = "cpu",
    )

    batch_size = 2
    seq_len = 5
    vocab_size = 100

    input_ids  = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size,seq_len)

    outputs = model(
        input_ids  =input_ids,
        attention_mask = attention_mask,
        labels = input_ids,
    )

    assert hasattr(outputs, "loss")
    assert outputs.loss.dim() == 0 # scalar loss


def test_olmo_causal_lm_deepchem_training_step():
    """Test that OLMoCausalLM works inside DeepChem TorchModel."""

    torch.manual_seed(0)
    np.random.seed(0)

    model= OLMoCausalLM(
        model_name = "sshleifer/tiny-gpt2",
        torch_dtype = torch.float32,
        device_map = "cpu",
    )

    # Fake tokenized dataset
    X = np.random.randint(0,50, size=(4, 6)).astype(np.int64)
    y = X.copy() # causal LM labels == imput_ids

    dataset  = dc.data.NumpyDataset(X, y)

    dc_model = dc.models.TorchModel(
        model = model,
        loss = None, # HF returns loss internally
        batch_size = 2,
    )

    loss = dc_model.fit(dataset, nb_epoch = 1)

    assert loss is not None