import os
import deepchem as dc
import tempfile
import numpy as np
import pytest
from deepchem.feat.molecule_featurizers import MXMNetFeaturizer

try:
    import torch
    from deepchem.models.torch_models.mxmnet import MXMNet
    has_torch = True
except:
    has_torch = False

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

    seed = 123
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = 'cpu'
    torch.set_default_device(device)

    dim = 10
    n_layer = 6
    cutoff = 5
    feat = MXMNetFeaturizer()

    loader = dc.data.SDFLoader(tasks=[QM9_TASKS[0]],
                               featurizer=feat,
                               sanitize=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "assets/qm9_mini.sdf")

    dataset = loader.create_dataset(inputs=dataset_path, shard_size=1)

    model = MXMNet(dim=dim, n_layer=n_layer, cutoff=cutoff)

    train_dir = None
    if train_dir is None:
        train_dir = tempfile.mkdtemp()
    data = dataset.select([i for i in range(1, 3)], train_dir)

    # prepare batch (size 2)
    data = data.X
    data = [data[i].to_pyg_graph() for i in range(2)]
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(data).to(device)

    model.to(device)
    output = model(pyg_batch)
    required_output = np.asarray([[-3.2702], [-2.9920]])
    assert np.allclose(output[0].cpu().detach().numpy(),
                       required_output[0],
                       atol=1e-04)
    assert np.allclose(output[1].cpu().detach().numpy(),
                       required_output[1],
                       atol=1e-04)
    assert output.shape == (2, 1)
