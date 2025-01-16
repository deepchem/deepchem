import pytest
import numpy as np
import deepchem as dc
try:
    import torch
except (ImportError, ModuleNotFoundError):
    pass

@pytest.mark.torch
def test_DAG_layer():
    """Test invoking DAGLayer."""
    np.random.seed(123)
    batch_size = 10
    n_graph_feat = 30
    n_atom_feat = 75
    max_atoms = 50
    layer_sizes = [100]
    atom_features = np.random.rand(batch_size, n_atom_feat)
    parents = np.random.randint(0,
                                max_atoms,
                                size=(batch_size, max_atoms, max_atoms))
    calculation_orders = np.random.randint(0,
                                           batch_size,
                                           size=(batch_size, max_atoms))
    calculation_masks = np.random.randint(0, 2, size=(batch_size, max_atoms))
    # Recall that the DAG layer expects a MultiConvMol as input,
    # so the "batch" is a pooled set of atoms from all the
    # molecules in the batch, just as it is for the graph conv.
    # This means that n_atoms is the batch-size
    n_atoms = batch_size
    layer = dc.models.torch_models.DAGLayer(n_graph_feat=n_graph_feat,
                            n_atom_feat=n_atom_feat,
                            max_atoms=max_atoms,
                            layer_sizes=layer_sizes)
    outputs = layer([  # noqa: F841
        atom_features,
        parents,
        calculation_orders,
        calculation_masks,
        np.array(n_atoms),
    ])
    # The output should be of shape (max_atom-th number of target atoms, n_graph_feat)
    assert outputs.shape == (6, 30) 


@pytest.mark.torch
def test_DAG_gather():
    """Test invoking DAGGather."""
    # TODO(rbharath): We need more documentation about why
    # these numbers work.
    np.random.seed(123)
    batch_size = 10
    n_graph_feat = 30
    n_atom_feat = 30
    n_outputs = 75
    max_atoms = 50
    layer_sizes = [100]
    layer = dc.models.torch_models.DAGGather(n_graph_feat=n_graph_feat,
                             n_outputs=n_outputs,
                             max_atoms=max_atoms,
                             layer_sizes=layer_sizes)
    atom_features = np.random.rand(batch_size, n_atom_feat)
    membership = np.sort(np.random.randint(0, batch_size, size=(batch_size)))
    outputs = layer([atom_features, membership])  # noqa: F841
    # The output should be of shape (membership.max()+1, n_outputs)
    assert outputs.shape == (10, 75)