import pytest
import deepchem as dc
from deepchem.feat.graph_data import BatchGraphData


@pytest.mark.torch
def test_grover_batch_mol_graph():
    import torch
    from deepchem.utils.grover import extract_grover_attributes
    grover_featurizer = dc.feat.GroverFeaturizer(
        features_generator=dc.feat.CircularFingerprint())

    smiles = ['CC', 'CCC']
    mol_graphs = grover_featurizer.featurize(smiles)
    mol_graph = mol_graphs[0]
    batched_mol_graph = BatchGraphData(mol_graphs)
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels, additional_features = extract_grover_attributes(
        batched_mol_graph)
    # 6 atoms: CC -> 2, CCC -> 3
    assert f_atoms.shape == (5, mol_graph.node_features.shape[1])
    # 7 bonds: CC -> 2, CCC -> 4 (bonds are considered as undirected
    # and a single bond contributes to 2 bonds)
    assert f_bonds.shape == (6, mol_graph.edge_features.shape[1])
    assert fg_labels.shape == (2, mol_graph.fg_labels.shape[0])
    assert additional_features.shape == (2,
                                         mol_graph.additional_features.shape[0])
    assert (a_scope == torch.Tensor([[0, 2], [2, 3]])).all()
    assert (b_scope == torch.Tensor([[0, 2], [2, 4]])).all()

    assert (a2b == torch.Tensor([[1, 0], [0, 0], [3, 0], [5, 0], [2, 4]])).all()
    assert (b2a == torch.Tensor([0, 1, 2, 4, 3, 4])).all()

    assert (b2revb == torch.Tensor([1, 0, 3, 2, 5, 4])).all()
    assert (a2a == torch.Tensor([[1, 0], [0, 0], [4, 0], [4, 0], [2, 3]])).all()
