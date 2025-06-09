import pytest
import numpy as np
import deepchem as dc

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_fno_overfit():
    from deepchem.models.torch_models import FNO
    model = FNO(input_dim=1, output_dim=1, modes=1, width=128, dims=1, depth=4)
    X = torch.rand(100, 100, 1)
    y = X
    dataset = dc.data.NumpyDataset(X=X,
                                   y=y)
    loss = model.fit(dataset, nb_epoch=300)
    assert loss < 1e-2, "Model can't overfit"

# @pytest.mark.torch
# def test_fno_heat_equation():
