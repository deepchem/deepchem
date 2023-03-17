try:
    import torch
except ModuleNotFoundError:
    pass


def testGroverEmbedding():
    from deepchem.models.torch_models.grover import GroverEmbedding
    hidden_size = 8
    f_atoms = torch.randn(4, 151)
    f_bonds = torch.randn(5, 165)
    a2b = torch.Tensor([[0, 0], [2, 0], [4, 0], [1, 3]]).type(torch.int32)
    b2a = torch.Tensor([0, 1, 3, 2, 3]).type(torch.long)
    b2revb = torch.Tensor([0, 2, 1, 4, 3]).type(torch.long)
    a_scope = torch.Tensor([[1, 3]]).type(torch.int32)
    b_scope = torch.Tensor([[1, 4]]).type(torch.int32)
    a2a = torch.Tensor([[0, 0], [3, 0], [3, 0], [1, 2]]).type(torch.int32)
    n_atoms, n_bonds = f_atoms.shape[0], f_bonds.shape[0]
    node_fdim, edge_fdim = f_atoms.shape[1], f_bonds.shape[1]
    layer = GroverEmbedding(edge_fdim=edge_fdim,
                            node_fdim=node_fdim,
                            hidden_size=8,
                            embedding_output_type='both')
    output = layer([f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a])
    assert output['atom_from_atom'].shape == (n_atoms, hidden_size)
    assert output['bond_from_atom'].shape == (n_bonds, hidden_size)
    assert output['atom_from_bond'].shape == (n_atoms, hidden_size)
    assert output['bond_from_bond'].shape == (n_bonds, hidden_size)
