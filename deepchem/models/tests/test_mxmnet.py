import torch
import numpy
from deepchem.models.torch_models.mxmnet import MXMNet
import deepchem as dc
import tempfile
import numpy as np
from deepchem.feat.molecule_featurizers import MXMNetFeaturizer


def test_mxmnet_regression():
    """
    Test MXMNet class for regression
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    feat = MXMNetFeaturizer()
    dataset_dc = dc.molnet.load_qm9(featurizer=feat)

    dim = 10
    n_layer = 6
    cutoff = 5
    model = MXMNet(dim=dim, n_layer=n_layer, cutoff=cutoff)

    tasks, dataset, transformers = dataset_dc
    train, valid, test = dataset
    train_dir = None
    if train_dir is None:
        train_dir = tempfile.mkdtemp()
    graph = train.select([i for i in range(1, 2)], train_dir)
    graph = dc.feat.GraphData(node_features=graph.X[0].node_features,
                              edge_index=graph.X[0].edge_index,
                              node_pos_features=graph.X[0].node_pos_features)
    data = graph.to_pyg_graph()

    model.to(device)
    output = model(data)
    required_output = np.asarray([-0.2781])
    assert np.allclose(output.detach().numpy(), required_output, atol=1e-04)
