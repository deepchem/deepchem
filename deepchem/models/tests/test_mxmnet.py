import deepchem as dc
import torch
import tempfile
import numpy as np
import pytest
from deepchem.feat.molecule_featurizers import MXMNetFeaturizer
from deepchem.models.torch_models.mxmnet import MXMNet

QM9_TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
    "h298", "g298"
]


@pytest.mark.torch
def test_mxmnet_regression():
    """
    Test MXMNet class for regression
    """
    try:
        from torch_geometric.data import Batch
    except ModuleNotFoundError:
        raise ImportError(
            "This test requires PyTorch Geometric to be installed.")

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    dim = 10
    n_layer = 6
    cutoff = 5
    feat = MXMNetFeaturizer()

    loader = dc.data.SDFLoader(tasks=[QM9_TASKS[0]],
                               featurizer=feat,
                               sanitize=True)

    dataset = loader.create_dataset(inputs="assets/qm9_mini.sdf", shard_size=1)

    model = MXMNet(dim=dim, n_layer=n_layer, cutoff=cutoff)

    train_dir = None
    if train_dir is None:
        train_dir = tempfile.mkdtemp()
    data = dataset.select([i for i in range(1, 3)], train_dir)

    # prepare batch (size 2)
    data = data.X
    data = [data[i].to_pyg_graph() for i in range(2)]
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(data)

    model.to(device)
    output = model(pyg_batch)
    required_output = np.asarray([0.0869, 0.1744])
    assert np.allclose(output[0].detach().numpy(),
                       required_output[0],
                       atol=1e-04)
    assert np.allclose(output[1].detach().numpy(),
                       required_output[1],
                       atol=1e-04)
