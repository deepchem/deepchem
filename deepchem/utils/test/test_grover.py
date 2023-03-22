import pytest
import deepchem as dc


@pytest.mark.torch
def test_grover_batch_mol_graph():
    import torch
    from deepchem.utils.grover import grover_batch_mol_graph
    grover_featurizer = dc.feat.GroverFeaturizer(
        features_generator=dc.feat.CircularFingerprint())

    smiles = ['CC', 'CCC']
    mol_graphs = grover_featurizer.featurize(smiles)
    mol_graph = mol_graphs[0]
    batched_mol_graphs = grover_batch_mol_graph(mol_graphs)
    assert batched_mol_graphs.smiles == smiles
    # 6 atoms: CC -> 2, CCC -> 3, 1 for padding
    assert batched_mol_graphs.f_atoms.shape == (
        6, mol_graph.node_features.shape[1])
    # 7 bonds: CC -> 2, CCC -> 4, 1 for padding (bonds are considered as undirected
    # and a single bond contributes to 2 bonds)
    assert batched_mol_graphs.f_bonds.shape == (
        7, mol_graph.edge_features.shape[1])
    assert batched_mol_graphs.fg_labels.shape == (2,
                                                  mol_graph.fg_labels.shape[0])
    assert batched_mol_graphs.additional_features.shape == (
        2, mol_graph.additional_features.shape[0])
    assert (batched_mol_graphs.a_scope == torch.Tensor([[1, 2], [3, 3]])).all()
    assert (batched_mol_graphs.b_scope == torch.Tensor([[1, 2], [3, 4]])).all()

    assert (batched_mol_graphs.a2b == torch.Tensor([[0, 0], [2, 0], [1, 0],
                                                    [4, 0], [6, 0],
                                                    [3, 5]])).all()
    assert (batched_mol_graphs.b2a == torch.Tensor([0, 1, 2, 3, 5, 4, 5])).all()
    assert (batched_mol_graphs.b2revb == torch.Tensor([0, 2, 1, 4, 3, 6,
                                                       5])).all()
    assert (batched_mol_graphs.a2a == torch.Tensor([[0, 0], [2, 0], [1, 0],
                                                    [5, 0], [5, 0],
                                                    [3, 4]])).all()
