import os
import deepchem as dc
import torch
import tempfile
import numpy as np
from deepchem.molnet.load_function.molnet_loader import _MolnetLoader
from deepchem.data import Dataset
from deepchem.feat.molecule_featurizers import MXMNetFeaturizer
from deepchem.models.torch_models.mxmnet import MXMNet

QM9_TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
    "h298", "g298"
]


class _QM9Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "datasets/qm9_sample.zip")
        if not os.path.exists(dataset_file):
            print("ulllu")
        loader = dc.data.SDFLoader(tasks=self.tasks,
                                   featurizer=self.featurizer,
                                   sanitize=True)
        return loader.create_dataset(dataset_file, shard_size=1)


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
    qm9 = _QM9Loader(featurizer=feat,
                     tasks=QM9_TASKS,
                     data_dir=None,
                     save_dir=None,
                     splitter='random',
                     transformer_generators=['normalization'])
    dataset_dc = qm9.load_dataset('qm9', reload=True)

    model = MXMNet(dim=dim, n_layer=n_layer, cutoff=cutoff)

    tasks, dataset, transformers = dataset_dc
    train, valid, test = dataset
    train_dir = None
    if train_dir is None:
        train_dir = tempfile.mkdtemp()
    data = train.select([i for i in range(1, 3)], train_dir)

    # prepare batch (size 2)
    data = data.X
    data = [data[i].to_pyg_graph() for i in range(2)]
    pyg_batch = Batch()
    pyg_batch = pyg_batch.from_data_list(data)

    model.to(device)
    output = model(pyg_batch)
    required_output = np.asarray([[-0.2781], [-0.4035]])
    assert np.allclose(output[0].detach().numpy(),
                       required_output[0],
                       atol=1e-04)
    assert np.allclose(output[1].detach().numpy(),
                       required_output[1],
                       atol=1e-04)
