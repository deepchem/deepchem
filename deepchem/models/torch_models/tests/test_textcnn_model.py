import pytest
import numpy as np
import deepchem as dc
try:
    import torch
    from deepchem.models.torch_models import TextCNNModel
    from deepchem.models.text_cnn import default_dict
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_textcnn_module():
    model = TextCNNModel(1, default_dict, 1)
    assert model.seq_length == max(model.kernel_sizes)
    large_length = 500
    model = TextCNNModel(1, default_dict, large_length)
    assert model.seq_length == large_length

@pytest.mark.torch
def test_forward_pass():
    smiles = np.array(['CC', 'CCC'])
    dataset = dc.data.NumpyDataset(X=smiles)
    char_dict, length = dc.models.TextCNNModel.build_char_dict(dataset)

    model = dc.models.TextCNNModel(len(delaney_tasks),
                                   char_dict,
                                   seq_length=length,
                                   mode='regression',
                                   learning_rate=1e-3,
                                   batch_size=batch_size,
                                   use_queue=False)
    