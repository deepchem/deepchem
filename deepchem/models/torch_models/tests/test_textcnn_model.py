import pytest
import numpy as np
try:
    import torch
    from deepchem.models.torch_models.text_cnn import default_dict, TextCNN
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_textcnn_base():
    model = TextCNN(1, default_dict, 1)
    assert model.seq_length == max(model.kernel_sizes)
    large_length = 500
    model = TextCNN(1, default_dict, large_length)
    assert model.seq_length == large_length


@pytest.mark.torch
def test_textcnn_base_forward():
    batch_size = 1
    input_tensor = torch.randint(34, (batch_size, 64))
    cls_model = TextCNN(1, default_dict, 1, mode="classification")
    reg_model = TextCNN(1, default_dict, 1, mode="regression")
    cls_output = cls_model.forward(input_tensor)
    reg_output = reg_model.forward(input_tensor)
    assert len(cls_output) == 2
    assert len(reg_output) == 1
    assert np.allclose(torch.sum(cls_output[0]).item(), 1, rtol=1e-5, atol=1e-6)
