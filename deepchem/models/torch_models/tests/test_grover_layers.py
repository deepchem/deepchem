import pytest

try:
    import torch
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def testGroverEmbedding(grover_batched_graph):
    from deepchem.models.torch_models.grover_layers import GroverEmbedding
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels = grover_batched_graph.get_components(
    )
    hidden_size = 8
    n_atoms, n_bonds = f_atoms.shape[0], f_bonds.shape[0]
    node_fdim, edge_fdim = f_atoms.shape[1], f_bonds.shape[1]
    layer = GroverEmbedding(hidden_size=hidden_size,
                            edge_fdim=edge_fdim,
                            node_fdim=node_fdim)
    output = layer([f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a])
    assert output['atom_from_atom'].shape == (n_atoms, hidden_size)
    assert output['bond_from_atom'].shape == (n_bonds, hidden_size)
    assert output['atom_from_bond'].shape == (n_atoms, hidden_size)
    assert output['bond_from_bond'].shape == (n_bonds, hidden_size)


@pytest.mark.torch
def testGroverBondVocabPredictor():
    from deepchem.models.torch_models.grover_layers import GroverBondVocabPredictor
    num_bonds = 20
    in_features, vocab_size = 16, 10
    layer = GroverBondVocabPredictor(vocab_size, in_features)
    embedding = torch.randn(num_bonds * 2, in_features)
    result = layer(embedding)
    assert result.shape == (num_bonds, vocab_size)


@pytest.mark.torch
def testGroverAtomVocabPredictor():
    from deepchem.models.torch_models.grover_layers import GroverAtomVocabPredictor
    num_atoms, in_features, vocab_size = 30, 16, 10
    layer = GroverAtomVocabPredictor(vocab_size, in_features)
    embedding = torch.randn(num_atoms, in_features)
    result = layer(embedding)
    assert result.shape == (num_atoms, vocab_size)


@pytest.mark.torch
def testGroverFunctionalGroupPredictor():
    from deepchem.models.torch_models.grover_layers import GroverFunctionalGroupPredictor
    in_features, functional_group_size = 8, 20
    num_atoms, num_bonds = 10, 20
    predictor = GroverFunctionalGroupPredictor(functional_group_size=20,
                                               in_features=8)
    # In a batched graph, atoms and bonds belonging to different graphs are differentiated
    # via scopes. In the below scenario, we assume a batched mol graph of three molecules
    # with 10 atoms, 20 bonds. On the 10 atoms, we consider the first 3 belonging to mol1,
    # next 3 belonging to mol2 and remaining 4 belonging to mol4.
    # Hence, the atom scope is [(0, 3), (3, 3), (6, 4)]. Similarly, for bonds, we have first 5 bonds belonging to mol1, next 4 to mol2 and remaining 11 to bond3.
    atom_scope, bond_scope = [(0, 3), (3, 3), (6, 4)], [(0, 5), (5, 4), (9, 11)]
    embeddings = {}
    embeddings['bond_from_atom'] = torch.randn(num_bonds, in_features)
    embeddings['bond_from_bond'] = torch.randn(num_bonds, in_features)
    embeddings['atom_from_atom'] = torch.randn(num_atoms, in_features)
    embeddings['atom_from_bond'] = torch.randn(num_atoms, in_features)

    result = predictor(embeddings, atom_scope, bond_scope)
    assert result['bond_from_bond'].shape == (len(bond_scope),
                                              functional_group_size)
    assert result['bond_from_atom'].shape == (len(bond_scope),
                                              functional_group_size)
    assert result['atom_from_atom'].shape == (len(atom_scope),
                                              functional_group_size)
    assert result['atom_from_bond'].shape == (len(atom_scope),
                                              functional_group_size)


@pytest.mark.torch
@pytest.mark.parametrize('dynamic_depth', ['none', 'uniform'])
@pytest.mark.parametrize('atom_messages', [False, True])
def testGroverMPNEncoder(grover_batched_graph, dynamic_depth, atom_messages):
    from deepchem.models.torch_models.grover_layers import GroverMPNEncoder
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, _, _, _ = grover_batched_graph.get_components(
    )

    # TODO Write tests for undirected = True case, currently fails. for this case, we have
    # to generate inputs (a2b, b2a, b2revb) for undirected graph.
    hidden_size = 32
    depth = 5
    undirected = False
    attach_feats = True
    if not atom_messages:
        init_message_dim = f_bonds.shape[1]
        attached_feat_fdim = f_atoms.shape[1]
        layer = GroverMPNEncoder(atom_messages=atom_messages,
                                 init_message_dim=init_message_dim,
                                 attached_feat_fdim=attached_feat_fdim,
                                 hidden_size=hidden_size,
                                 depth=depth,
                                 dynamic_depth=dynamic_depth,
                                 undirected=undirected,
                                 attach_feats=attach_feats)
        init_messages = f_bonds
        init_attached_features = f_atoms
        a2nei = a2b
        a2attached = a2a
        out = layer(init_messages, init_attached_features, a2nei, a2attached,
                    b2a, b2revb)
        assert out.shape == (f_bonds.shape[0], hidden_size)
    else:
        init_message_dim = f_atoms.shape[1]
        attached_feat_fdim = f_bonds.shape[1]

        layer = GroverMPNEncoder(atom_messages=atom_messages,
                                 init_message_dim=init_message_dim,
                                 attached_feat_fdim=attached_feat_fdim,
                                 hidden_size=hidden_size,
                                 depth=depth,
                                 dynamic_depth=dynamic_depth,
                                 undirected=undirected,
                                 attach_feats=attach_feats)
        init_attached_features = f_bonds
        init_messages = f_atoms
        a2nei = a2a
        a2attached = a2b
        out = layer(init_messages, init_attached_features, a2nei, a2attached,
                    b2a, b2revb)
        assert out.shape == (f_atoms.shape[0], hidden_size)


@pytest.mark.torch
def testGroverAttentionHead(grover_batched_graph):
    from deepchem.models.torch_models.grover_layers import GroverAttentionHead
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, _, _, _ = grover_batched_graph.get_components(
    )
    hidden_size = 165
    atom_messages = False
    layer = GroverAttentionHead(hidden_size,
                                bias=True,
                                depth=4,
                                undirected=False,
                                atom_messages=atom_messages)
    query, key, value = layer(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
    assert query.size() == (f_bonds.shape[0], hidden_size)
    assert key.size() == (f_bonds.shape[0], hidden_size)
    assert value.size() == (f_bonds.shape[0], hidden_size)


@pytest.mark.torch
def testGroverMTBlock(grover_batched_graph):
    from deepchem.models.torch_models.grover_layers import GroverMTBlock
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _ = grover_batched_graph.get_components(
    )

    hidden_size = 16
    layer = GroverMTBlock(atom_messages=True,
                          input_dim=f_atoms.shape[1],
                          num_heads=4,
                          depth=1,
                          hidden_size=hidden_size)

    new_batch = layer(
        [f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a])
    new_f_atoms, new_f_bonds, new_a2b, new_b2a, new_b2revb, new_a_scope, new_b_scope, new_a2a = new_batch
    # The shapes should match the earlier shapes because message passing only updates node features.
    assert new_f_atoms.shape == (f_atoms.shape[0], hidden_size)
    assert new_f_bonds.shape == f_bonds.shape
    # The following variables are utility variables used during message passing to compute neighbors. Here we are asserting that MTBlock layer is not modifying these variables.
    assert (new_a2b == a2b).all()
    assert (new_b2a == b2a).all()
    assert (new_b2revb == b2revb).all()
    assert (new_a_scope == a_scope).all()
    assert (new_b_scope == b_scope).all()
    assert (new_a2a == a2a).all()


@pytest.mark.torch
def testGroverTransEncoder(grover_batched_graph):
    from deepchem.models.torch_models.grover_layers import GroverTransEncoder
    f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, _ = grover_batched_graph.get_components(
    )

    hidden_size = 8
    n_atoms, n_bonds = f_atoms.shape[0], f_bonds.shape[0]
    node_fdim, edge_fdim = f_atoms.shape[1], f_bonds.shape[1]
    layer = GroverTransEncoder(hidden_size=hidden_size,
                               edge_fdim=edge_fdim,
                               node_fdim=node_fdim)
    output = layer([f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a])
    assert output[0][0].shape == (n_atoms, hidden_size)
    assert output[0][1].shape == (n_bonds, hidden_size)
    assert output[1][0].shape == (n_atoms, hidden_size)
    assert output[1][1].shape == (n_bonds, hidden_size)
