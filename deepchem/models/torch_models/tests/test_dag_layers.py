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


@pytest.mark.torch
def test_DAG_layer_correctness():
    """Test that torch DAGLayer matches TF DAGLayer output."""
    from deepchem.models.torch_models import DAGLayer
    np.random.seed(123)
    batch_size = 10
    n_graph_feat = 3
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
    n_atoms = batch_size

    # choosing a random column from the output of the tensorflow DAGLayer (1st column)
    expected_outputs = torch.tensor([
        6.7915114e+11, 9.3723308e+13, 1.2509578e+10, 8.4379546e+11,
        1.9989182e+16, 0.0000000e+00
    ],
                                    dtype=torch.float32)

    dag_layer = DAGLayer(n_graph_feat=n_graph_feat,
                         n_atom_feat=n_atom_feat,
                         max_atoms=max_atoms,
                         layer_sizes=layer_sizes)

    # Create numpy arrays with matching shapes and dtypes
    new_w0 = np.random.normal(size=dag_layer.W_layers[0].shape).astype(
        np.float32)
    new_b0 = np.zeros(dag_layer.b_layers[0].shape).astype(np.float32)
    new_w1 = np.random.normal(size=dag_layer.W_layers[1].shape).astype(
        np.float32)
    new_b1 = np.zeros(dag_layer.b_layers[1].shape).astype(np.float32)

    # Assign the new weights
    with torch.no_grad():
        dag_layer.W_layers[0].copy_(torch.from_numpy(new_w0))
        dag_layer.b_layers[0].copy_(torch.from_numpy(new_b0))
        dag_layer.W_layers[1].copy_(torch.from_numpy(new_w1))
        dag_layer.b_layers[1].copy_(torch.from_numpy(new_b1))
    dag_outputs = dag_layer([
        atom_features, parents, calculation_orders, calculation_masks,
        np.array(n_atoms)
    ])
    assert torch.allclose(
        dag_outputs[:, 1], expected_outputs,
        atol=1e-6), "Outputs from TF and Torch W_layers do not match!"


@pytest.mark.torch
def test_DAG_gather_correctness():
    """Test that torch DAGather matches TF DAGgather output."""
    np.random.seed(123)
    batch_size = 10
    n_graph_feat = 30
    n_atom_feat = 30
    max_atoms = 50
    n_outputs = 3
    layer_sizes = [100]
    atom_features = np.random.rand(batch_size, n_atom_feat)
    membership = np.sort(np.random.randint(0, batch_size, size=(batch_size)))
    # choosing a random column from the output of the tensorflow DAGGather (1st column)
    expected_outputs = torch.tensor([
        0.0000, 0.0000, 5.760246, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        14.064488, 15.860514
    ],
                                    dtype=torch.float32)

    dag_gather = dc.models.torch_models.DAGGather(n_graph_feat=n_graph_feat,
                                                  n_outputs=n_outputs,
                                                  max_atoms=max_atoms,
                                                  layer_sizes=layer_sizes)

    # Create numpy arrays with matching shapes and dtypes
    new_w0 = np.random.normal(size=dag_gather.W_layers[0].shape).astype(
        np.float32)
    new_b0 = np.zeros(dag_gather.b_layers[0].shape).astype(np.float32)
    new_w1 = np.random.normal(size=dag_gather.W_layers[1].shape).astype(
        np.float32)
    new_b1 = np.zeros(dag_gather.b_layers[1].shape).astype(np.float32)

    # Assign the new weights
    with torch.no_grad():
        dag_gather.W_layers[0].copy_(torch.from_numpy(new_w0))
        dag_gather.b_layers[0].copy_(torch.from_numpy(new_b0))
        dag_gather.W_layers[1].copy_(torch.from_numpy(new_w1))
        dag_gather.b_layers[1].copy_(torch.from_numpy(new_b1))
    dag_outputs = dag_gather([atom_features, membership])
    assert torch.allclose(
        dag_outputs[:, 1], expected_outputs,
        atol=1e-6), "Outputs from TF and Torch DAGGather do not match!"
