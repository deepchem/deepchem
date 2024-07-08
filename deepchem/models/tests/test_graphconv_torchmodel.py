import deepchem as dc
import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
    from deepchem.models.torch_models import _GraphConvTorchModel
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_graph_conv_classification():
    batch_size = 10
    out_channels = 2
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    atom_features = torch.from_numpy(multi_mol.get_atom_features().astype(
        np.float32))
    degree_slice = torch.from_numpy(multi_mol.deg_slice)
    membership = torch.from_numpy(multi_mol.membership)
    deg_adjs = [
        torch.from_numpy(i) for i in multi_mol.get_deg_adjacency_lists()[1:]
    ]
    args = [atom_features, degree_slice, membership, torch.tensor(2)] + deg_adjs
    model_p = _GraphConvTorchModel(
        out_channels,
        graph_conv_layers=[64, 64],
        number_input_features=[atom_features.shape[-1], 64],
        dense_layer_size=128,
        dropout=0.0,
        mode="classification",
        number_atom_features=75,
        n_classes=2,
        batch_normalize=False,
        uncertainty=False,
        batch_size=batch_size)
    W_list = np.load("deepchem/models/tests/assets/graphconvlayer0_weights.npy",
                     allow_pickle=True).tolist()
    model_p.graph_convs[0].W_list = nn.ParameterList(
        [nn.Parameter(torch.tensor(k)) for k in W_list])
    b_list = np.load("deepchem/models/tests/assets/graphconvlayer0_biases.npy",
                     allow_pickle=True).tolist()
    model_p.graph_convs[0].b_list = nn.ParameterList(
        [nn.Parameter(torch.tensor(k)) for k in b_list])
    W_list = np.load("deepchem/models/tests/assets/graphconvlayer1_weights.npy",
                     allow_pickle=True).tolist()
    model_p.graph_convs[1].W_list = nn.ParameterList(
        [nn.Parameter(torch.tensor(k)) for k in W_list])
    b_list = np.load("deepchem/models/tests/assets/graphconvlayer1_biases.npy",
                     allow_pickle=True).tolist()
    model_p.graph_convs[1].b_list = nn.ParameterList(
        [nn.Parameter(torch.tensor(k)) for k in b_list])

    dense_weights = np.load("deepchem/models/tests/assets/dense_weights.npy")
    dense_biases = np.load("deepchem/models/tests/assets/dense_biases.npy")
    model_p.dense.weight.data = torch.from_numpy(np.transpose(dense_weights))
    model_p.dense.bias.data = torch.from_numpy(dense_biases)

    reshapedense_weights = np.load(
        "deepchem/models/tests/assets/reshapedense_weights.npy")
    reshapedense_biases = np.load(
        "deepchem/models/tests/assets/reshapedense_biases.npy")
    model_p.reshape_dense.weight.data = torch.from_numpy(
        np.transpose(reshapedense_weights))
    model_p.reshape_dense.bias.data = torch.from_numpy(reshapedense_biases)

    result_p = model_p(args)
    assert np.allclose(
        result_p[0].detach().numpy(),
        np.load(
            "deepchem/models/tests/assets/graphconvmodel_output_classification.npy"
        ),
        atol=1e-4)
    assert np.allclose(
        result_p[1].detach().numpy(),
        np.load(
            "deepchem/models/tests/assets/graphconvmodel_logits_classification.npy"
        ),
        atol=1e-4)
    assert np.allclose(
        result_p[2].detach().numpy(),
        np.load(
            "deepchem/models/tests/assets/graphconvmodel_neural_classification.npy"
        ),
        atol=1e-4)
    assert len(result_p) == 3
