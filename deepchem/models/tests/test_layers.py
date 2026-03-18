import deepchem as dc
import numpy as np
import pytest
import os

try:
    import tensorflow as tf
    import deepchem.models.layers as layers
    from tensorflow.python.framework import test_util  # noqa: F401
    has_tensorflow = True
except ModuleNotFoundError:
    has_tensorflow = False

try:
    import torch
    import torch.nn as nn
    from deepchem.models.torch_models import layers as torch_layers
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.tensorflow
def test_cosine_dist():
    """Test invoking cosine_dist."""
    x = tf.ones((5, 4), dtype=tf.dtypes.float32, name=None)
    y_same = tf.ones((5, 4), dtype=tf.dtypes.float32, name=None)
    # x and y are the same tensor (equivalent at every element)
    # the pairwise inner product of the rows in x and y will always be 1
    # the output tensor will be of shape (5,5)
    cos_sim_same = layers.cosine_dist(x, y_same)
    diff = cos_sim_same - tf.ones((5, 5), dtype=tf.dtypes.float32, name=None)
    assert tf.abs(tf.reduce_sum(diff)) < 1e-5  # True

    identity_tensor = tf.eye(
        512, dtype=tf.dtypes.float32)  # identity matrix of shape (512,512)
    x1 = identity_tensor[0:256, :]
    x2 = identity_tensor[256:512, :]
    # each row in x1 is orthogonal to each row in x2
    # the pairwise inner product of the rows in x and y will always be 0
    # the output tensor will be of shape (256,256)
    cos_sim_orth = layers.cosine_dist(x1, x2)
    assert tf.abs(tf.reduce_sum(cos_sim_orth)) < 1e-5  # True
    assert all([cos_sim_orth.shape[dim] == 256 for dim in range(2)])  # True


@pytest.mark.tensorflow
def test_highway():
    """Test invoking Highway."""
    width = 5
    batch_size = 10
    input = np.random.rand(batch_size, width).astype(np.float32)
    layer = layers.Highway()
    result = layer(input)
    assert result.shape == (batch_size, width)
    assert len(layer.trainable_variables) == 4

    # Creating a second layer should produce different results, since it has
    # different random weights.

    layer2 = layers.Highway()
    result2 = layer2(input)
    assert not np.allclose(result, result2)

    # But evaluating the first layer again should produce the same result as before.

    result3 = layer(input)
    assert np.allclose(result, result3)


@pytest.mark.tensorflow
def test_combine_mean_std():
    """Test invoking CombineMeanStd."""
    mean = np.random.rand(5, 3).astype(np.float32)
    std = np.random.rand(5, 3).astype(np.float32)
    layer = layers.CombineMeanStd(training_only=True, noise_epsilon=0.01)
    result1 = layer([mean, std], training=False)
    assert np.array_equal(result1, mean)  # No noise in test mode
    result2 = layer([mean, std], training=True)
    assert not np.array_equal(result2, mean)
    assert np.allclose(result2, mean, atol=0.1)


@pytest.mark.tensorflow
def test_stack():
    """Test invoking Stack."""
    input1 = np.random.rand(5, 4).astype(np.float32)
    input2 = np.random.rand(5, 4).astype(np.float32)
    result = layers.Stack()([input1, input2])
    assert result.shape == (5, 2, 4)
    assert np.array_equal(input1, result[:, 0, :])
    assert np.array_equal(input2, result[:, 1, :])


@pytest.mark.tensorflow
def test_variable():
    """Test invoking Variable."""
    value = np.random.rand(5, 4).astype(np.float32)
    layer = layers.Variable(value)
    layer.build([])
    result = layer.call([]).numpy()
    assert np.allclose(result, value)
    assert len(layer.trainable_variables) == 1


@pytest.mark.tensorflow
def test_interatomic_l2_distances():
    """Test invoking InteratomicL2Distances."""
    atoms = 5
    neighbors = 2
    coords = np.random.rand(atoms, 3)
    neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
    layer = layers.InteratomicL2Distances(atoms, neighbors, 3)
    result = layer([coords, neighbor_list])
    assert result.shape == (atoms, neighbors)
    for atom in range(atoms):
        for neighbor in range(neighbors):
            delta = coords[atom] - coords[neighbor_list[atom, neighbor]]
            dist2 = np.dot(delta, delta)
            assert np.allclose(dist2, result[atom, neighbor])


@pytest.mark.tensorflow
def test_weave_layer():
    """Test invoking WeaveLayer."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    weave = layers.WeaveLayer(init=tf.keras.initializers.TruncatedNormal(
        stddev=0.03))
    atom_feat = []
    pair_feat = []
    atom_to_pair = []
    pair_split = []
    start = 0
    n_pair_feat = 14
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        # index of pair features
        C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
        atom_to_pair.append(
            np.transpose(np.array([C1.flatten() + start,
                                   C0.flatten() + start])))
        # number of pairs for each atom
        pair_split.extend(C1.flatten() + start)
        start = start + n_atoms

        # atom features
        atom_feat.append(mol.get_atom_features())
        # pair features
        pair_feat.append(
            np.reshape(mol.get_pair_features(),
                       (n_atoms * n_atoms, n_pair_feat)))
    inputs = [
        np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
        np.concatenate(pair_feat, axis=0),
        np.array(pair_split),
        np.concatenate(atom_to_pair, axis=0)
    ]
    # Outputs should be [A, P]
    outputs = weave(inputs)
    assert len(outputs) == 2


@pytest.mark.tensorflow
def test_weave_gather():
    """Test invoking WeaveGather."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    atom_feat = []
    atom_split = []
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        atom_split.extend([im] * n_atoms)

        # atom features
        atom_feat.append(mol.get_atom_features())
    inputs = [
        np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
        np.array(atom_split)
    ]
    # Try without compression
    gather = layers.WeaveGather(batch_size=2, n_input=75, gaussian_expand=True)
    # Outputs should be [mol1_vec, mol2_vec)
    outputs = gather(inputs)
    assert len(outputs) == 2
    assert np.array(outputs[0]).shape == (11 * 75,)
    assert np.array(outputs[1]).shape == (11 * 75,)

    # Try with compression
    gather = layers.WeaveGather(batch_size=2,
                                n_input=75,
                                gaussian_expand=True,
                                compress_post_gaussian_expansion=True)
    # Outputs should be [mol1_vec, mol2_vec)
    outputs = gather(inputs)
    assert len(outputs) == 2
    assert np.array(outputs[0]).shape == (75,)
    assert np.array(outputs[1]).shape == (75,)


@pytest.mark.tensorflow
def test_weave_gather_gaussian_histogram():
    """Test Gaussian Histograms."""
    from rdkit import Chem
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    gather = layers.WeaveGather(batch_size=2, n_input=75)
    atom_feat = []
    atom_split = []
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        atom_split.extend([im] * n_atoms)

        # atom features
        atom_feat.append(mol.get_atom_features())
    inputs = [
        np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
        np.array(atom_split)
    ]
    # per_mol_features = tf.math.segment_sum(inputs[0], inputs[1])
    outputs = gather.gaussian_histogram(inputs[0])
    # Gaussian histograms expands into 11 Gaussian buckets.
    assert np.array(outputs).shape == (
        4,
        11 * 75,
    )
    # assert np.array(outputs[1]).shape == (11 * 75,)


@pytest.mark.tensorflow
def test_graph_conv():
    """Test invoking GraphConv."""
    out_channels = 2
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    atom_features = multi_mol.get_atom_features().astype(np.float32)
    degree_slice = multi_mol.deg_slice
    membership = multi_mol.membership
    deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
    args = [atom_features, degree_slice, membership] + deg_adjs
    layer = layers.GraphConv(out_channels)
    result = layer(args)
    assert result.shape == (n_atoms, out_channels)
    num_deg = 2 * layer.max_degree + (1 - layer.min_degree)
    assert len(layer.trainable_variables) == 2 * num_deg


@pytest.mark.tensorflow
def test_graph_pool():
    """Test invoking GraphPool."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    atom_features = multi_mol.get_atom_features().astype(np.float32)
    degree_slice = multi_mol.deg_slice
    membership = multi_mol.membership
    deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
    args = [atom_features, degree_slice, membership] + deg_adjs
    result = layers.GraphPool()(args)
    assert result.shape[0] == n_atoms
    # TODO What should shape[1] be?  It's not documented.


@pytest.mark.tensorflow
def test_graph_gather():
    """Test invoking GraphGather."""
    batch_size = 2
    n_features = 75
    # n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mols = featurizer.featurize(mols)
    multi_mol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(mols)
    atom_features = multi_mol.get_atom_features().astype(np.float32)
    degree_slice = multi_mol.deg_slice
    membership = multi_mol.membership
    deg_adjs = multi_mol.get_deg_adjacency_lists()[1:]
    args = [atom_features, degree_slice, membership] + deg_adjs
    result = layers.GraphGather(batch_size)(args)
    # TODO(rbharath): Why is it 2*n_features instead of n_features?
    assert result.shape == (batch_size, 2 * n_features)


@pytest.mark.tensorflow
def test_lstm_step():
    """Test invoking LSTMStep."""
    n_test = 5
    n_feat = 10
    y = np.random.rand(n_test, 2 * n_feat).astype(np.float32)
    state_zero = np.random.rand(n_test, n_feat).astype(np.float32)
    state_one = np.random.rand(n_test, n_feat).astype(np.float32)
    layer = layers.LSTMStep(n_feat, 2 * n_feat)
    result = layer([y, state_zero, state_one])
    h_out, h_copy_out, c_out = (result[0], result[1][0], result[1][1])
    assert h_out.shape == (n_test, n_feat)
    assert h_copy_out.shape == (n_test, n_feat)
    assert c_out.shape == (n_test, n_feat)
    assert len(layer.trainable_variables) == 1


@pytest.mark.tensorflow
def test_attn_lstm_embedding():
    """Test invoking AttnLSTMEmbedding."""
    max_depth = 5
    n_test = 5
    n_support = 11
    n_feat = 10
    test = np.random.rand(n_test, n_feat).astype(np.float32)
    support = np.random.rand(n_support, n_feat).astype(np.float32)
    layer = layers.AttnLSTMEmbedding(n_test, n_support, n_feat, max_depth)
    test_out, support_out = layer([test, support])
    assert test_out.shape == (n_test, n_feat)
    assert support_out.shape == (n_support, n_feat)
    assert len(layer.trainable_variables) == 4


@pytest.mark.tensorflow
def test_iter_ref_lstm_embedding():
    """Test invoking IterRefLSTMEmbedding."""
    max_depth = 5
    n_test = 5
    n_support = 11
    n_feat = 10
    test = np.random.rand(n_test, n_feat).astype(np.float32)
    support = np.random.rand(n_support, n_feat).astype(np.float32)
    layer = layers.IterRefLSTMEmbedding(n_test, n_support, n_feat, max_depth)
    test_out, support_out = layer([test, support])
    assert test_out.shape == (n_test, n_feat)
    assert support_out.shape == (n_support, n_feat)
    assert len(layer.trainable_variables) == 8


@pytest.mark.tensorflow
def test_vina_free_energy():
    """Test invoking VinaFreeEnergy."""
    n_atoms = 5
    m_nbrs = 1
    ndim = 3
    nbr_cutoff = 1
    start = 0
    stop = 4
    X = np.random.rand(n_atoms, ndim).astype(np.float32)
    Z = np.random.randint(0, 2, (n_atoms)).astype(np.float32)
    layer = layers.VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start,
                                  stop)
    result = layer([X, Z])
    assert len(layer.trainable_variables) == 6
    assert result.shape == tuple()

    # Creating a second layer should produce different results, since it has
    # different random weights.

    layer2 = layers.VinaFreeEnergy(n_atoms, m_nbrs, ndim, nbr_cutoff, start,
                                   stop)
    result2 = layer2([X, Z])
    assert not np.allclose(result, result2)

    # But evaluating the first layer again should produce the same result as before.

    result3 = layer([X, Z])
    assert np.allclose(result, result3)


@pytest.mark.tensorflow
def test_weighted_linear_combo():
    """Test invoking WeightedLinearCombo."""
    input1 = np.random.rand(5, 10).astype(np.float32)
    input2 = np.random.rand(5, 10).astype(np.float32)
    layer = layers.WeightedLinearCombo()
    result = layer([input1, input2])
    assert len(layer.trainable_variables) == 2
    expected = input1 * layer.trainable_variables[
        0] + input2 * layer.trainable_variables[1]
    assert np.allclose(result, expected)


@pytest.mark.tensorflow
def test_neighbor_list():
    """Test invoking NeighborList."""
    N_atoms = 5
    start = 0
    stop = 12
    nbr_cutoff = 3
    ndim = 3
    M_nbrs = 2
    coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
    coords = tf.cast(tf.stack(coords), tf.float32)
    layer = layers.NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start, stop)
    result = layer(coords)
    assert result.shape == (N_atoms, M_nbrs)


@pytest.mark.tensorflow
def test_atomic_convolution():
    """Test invoking AtomicConvolution."""
    batch_size = 4
    max_atoms = 5
    max_neighbors = 2
    dimensions = 3
    params = [[5.0, 2.0, 0.5], [10.0, 2.0, 0.5]]
    input1 = np.random.rand(batch_size, max_atoms,
                            dimensions).astype(np.float32)
    input2 = np.random.randint(max_atoms,
                               size=(batch_size, max_atoms, max_neighbors))
    input3 = np.random.randint(1,
                               10,
                               size=(batch_size, max_atoms, max_neighbors))
    layer = layers.AtomicConvolution(radial_params=params)
    result = layer([input1, input2, input3])
    assert result.shape == (batch_size, max_atoms, len(params))
    assert len(layer.trainable_variables) == 3


@pytest.mark.tensorflow
def test_alpha_share_layer():
    """Test invoking AlphaShareLayer."""
    batch_size = 10
    length = 6
    input1 = np.random.rand(batch_size, length).astype(np.float32)
    input2 = np.random.rand(batch_size, length).astype(np.float32)
    layer = layers.AlphaShareLayer()
    result = layer([input1, input2])
    assert input1.shape == result[0].shape
    assert input2.shape == result[1].shape

    # Creating a second layer should produce different results, since it has
    # different random weights.

    layer2 = layers.AlphaShareLayer()
    result2 = layer2([input1, input2])
    assert not np.allclose(result[0], result2[0])
    assert not np.allclose(result[1], result2[1])

    # But evaluating the first layer again should produce the same result as before.

    result3 = layer([input1, input2])
    assert np.allclose(result[0], result3[0])
    assert np.allclose(result[1], result3[1])


@pytest.mark.tensorflow
def test_sluice_loss():
    """Test invoking SluiceLoss."""
    input1 = np.ones((3, 4)).astype(np.float32)
    input2 = np.ones((2, 2)).astype(np.float32)
    result = layers.SluiceLoss()([input1, input2])
    assert np.allclose(result, 40.0)


@pytest.mark.tensorflow
def test_beta_share():
    """Test invoking BetaShare."""
    batch_size = 10
    length = 6
    input1 = np.random.rand(batch_size, length).astype(np.float32)
    input2 = np.random.rand(batch_size, length).astype(np.float32)
    layer = layers.BetaShare()
    result = layer([input1, input2])
    assert input1.shape == result.shape
    assert input2.shape == result.shape

    # Creating a second layer should produce different results, since it has
    # different random weights.

    layer2 = layers.BetaShare()
    result2 = layer2([input1, input2])
    assert not np.allclose(result, result2)

    # But evaluating the first layer again should produce the same result as before.

    result3 = layer([input1, input2])
    assert np.allclose(result, result3)


@pytest.mark.tensorflow
def test_ani_feat():
    """Test invoking ANIFeat."""
    batch_size = 10
    max_atoms = 5
    input = np.random.rand(batch_size, max_atoms, 4).astype(np.float32)
    layer = layers.ANIFeat(max_atoms=max_atoms)
    result = layer(input)  # noqa: F841
    # TODO What should the output shape be?  It's not documented, and there
    # are no other test cases for it.


@pytest.mark.tensorflow
def test_graph_embed_pool_layer():
    """Test invoking GraphEmbedPoolLayer."""
    V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
    adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
    layer = layers.GraphEmbedPoolLayer(num_vertices=6)
    result = layer([V, adjs])
    assert result[0].shape == (10, 6, 50)
    assert result[1].shape == (10, 6, 5, 6)

    # Creating a second layer should produce different results, since it has
    # different random weights.

    layer2 = layers.GraphEmbedPoolLayer(num_vertices=6)
    result2 = layer2([V, adjs])
    assert not np.allclose(result[0], result2[0])
    assert not np.allclose(result[1], result2[1])

    # But evaluating the first layer again should produce the same result as before.

    result3 = layer([V, adjs])
    assert np.allclose(result[0], result3[0])
    assert np.allclose(result[1], result3[1])


@pytest.mark.tensorflow
def test_graph_cnn():
    """Test invoking GraphCNN."""
    V = np.random.uniform(size=(10, 100, 50)).astype(np.float32)
    adjs = np.random.uniform(size=(10, 100, 5, 100)).astype(np.float32)
    layer = layers.GraphCNN(num_filters=6)
    result = layer([V, adjs])
    assert result.shape == (10, 100, 6)

    # Creating a second layer should produce different results, since it has
    # different random weights.

    layer2 = layers.GraphCNN(num_filters=6)
    result2 = layer2([V, adjs])
    assert not np.allclose(result, result2)

    # But evaluating the first layer again should produce the same result as before.

    result3 = layer([V, adjs])
    assert np.allclose(result, result3)


@pytest.mark.tensorflow
def test_DAG_layer():
    """Test invoking DAGLayer."""
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
    # dropout_switch = False
    layer = layers.DAGLayer(n_graph_feat=n_graph_feat,
                            n_atom_feat=n_atom_feat,
                            max_atoms=max_atoms,
                            layer_sizes=layer_sizes)
    outputs = layer([  # noqa: F841
        atom_features,
        parents,
        calculation_orders,
        calculation_masks,
        n_atoms,
        # dropout_switch
    ])
    # TODO(rbharath): What is the shape of outputs supposed to be?
    # I'm getting (7, 30) here. Where does 7 come from??


@pytest.mark.tensorflow
def test_DAG_gather():
    """Test invoking DAGGather."""
    # TODO(rbharath): We need more documentation about why
    # these numbers work.
    batch_size = 10
    n_graph_feat = 30
    n_atom_feat = 30
    n_outputs = 75
    max_atoms = 50
    layer_sizes = [100]
    layer = layers.DAGGather(n_graph_feat=n_graph_feat,
                             n_outputs=n_outputs,
                             max_atoms=max_atoms,
                             layer_sizes=layer_sizes)
    atom_features = np.random.rand(batch_size, n_atom_feat)
    membership = np.sort(np.random.randint(0, batch_size, size=(batch_size)))
    outputs = layer([atom_features, membership])  # noqa: F841


@pytest.mark.torch
def test_scale_norm():
    """Test invoking ScaleNorm."""
    input_ar = torch.tensor([[1., 99., 10000.], [0.003, 999.37, 23.]])
    layer = torch_layers.ScaleNorm(0.35)
    result1 = layer(input_ar)
    output_ar = torch.tensor([[5.9157897e-05, 5.8566318e-03, 5.9157896e-01],
                              [1.7754727e-06, 5.9145141e-01, 1.3611957e-02]])
    assert torch.allclose(result1, output_ar)


@pytest.mark.torch
def test_multi_headed_mat_attention():
    """Test invoking MultiHeadedMATAttention."""
    feat = dc.feat.MATFeaturizer()
    input_smile = "CC"
    out = feat.featurize(input_smile)
    node = torch.tensor(out[0].node_features).float().unsqueeze(0)
    adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
    dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
    mask = torch.sum(torch.abs(node), dim=-1) != 0
    layer = torch_layers.MultiHeadedMATAttention(dist_kernel='softmax',
                                                 lambda_attention=0.33,
                                                 lambda_distance=0.33,
                                                 h=16,
                                                 hsize=1024,
                                                 dropout_p=0.0)
    op = torch_layers.MATEmbedding()(node)
    output = layer(op, op, op, mask, adj, dist)
    assert (output.shape == (1, 3, 1024))


@pytest.mark.torch
def test_position_wise_feed_forward():
    """Test invoking PositionwiseFeedForward."""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    layer = torch_layers.PositionwiseFeedForward(d_input=2,
                                                 d_hidden=2,
                                                 d_output=2,
                                                 activation='relu',
                                                 n_layers=1,
                                                 dropout_p=0.0)
    result = layer(input_ar)
    output_ar = torch.tensor([[0.4810, 0.0000], [1.9771, 0.0000]])
    assert torch.allclose(result, output_ar, rtol=1e-4)


@pytest.mark.torch
@pytest.mark.parametrize('skip_connection,batch_norm,expected',
                         [(False, False, [[0.2795, 0.4243], [0.2795, 0.4243]]),
                          (True, False, [[-0.9612, 2.3846], [-4.1104, 5.7606]]),
                          (False, True, [[0.2795, 0.4243], [0.2795, 0.4243]]),
                          (True, True, [[-0.9612, 2.3846], [-4.1104, 5.7606]])])
def test_multilayer_perceptron(skip_connection, batch_norm, expected):
    """Test invoking MLP."""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    layer = torch_layers.MultilayerPerceptron(d_input=2,
                                              d_output=2,
                                              d_hidden=(2, 2),
                                              activation_fn='relu',
                                              dropout=0.0,
                                              batch_norm=batch_norm,
                                              skip_connection=skip_connection)
    result = layer(input_ar)
    output_ar = torch.tensor(expected)
    assert torch.allclose(result, output_ar, atol=1e-4)


@pytest.mark.torch
def test_multilayer_perceptron_overfit():
    import torch
    import deepchem.models.torch_models.layers as torch_layers
    from deepchem.data import NumpyDataset
    from deepchem.models.torch_models.torch_model import TorchModel
    from deepchem.models.losses import L1Loss
    import numpy as np

    torch.manual_seed(0)
    x = torch.randn(10, 10)
    y = torch.ones(10, 1)
    data = NumpyDataset(x, y)
    layer = torch_layers.MultilayerPerceptron(d_input=10,
                                              d_output=1,
                                              d_hidden=(2, 2),
                                              activation_fn='relu')
    model = TorchModel(layer, loss=L1Loss())
    model.fit(data, nb_epoch=1000)
    output = model.predict_on_batch(data.X)
    assert np.allclose(output, y, atol=1e-2)


@pytest.mark.torch
def test_weighted_skip_multilayer_perceptron():
    "Test for weighted skip connection from the input to the output"
    seed = 123
    torch.manual_seed(seed)
    dim = 1
    features = torch.Tensor([[0.8343], [1.2713], [1.2713], [1.2713], [1.2713]])
    layer = dc.models.torch_models.layers.MultilayerPerceptron(
        d_input=dim,
        d_hidden=(dim,),
        d_output=dim,
        activation_fn='silu',
        skip_connection=True,
        weighted_skip=False)
    output = layer(features)
    output = output.detach().numpy()
    result = np.array([[1.1032], [1.5598], [1.5598], [1.5598], [1.5598]])
    assert np.allclose(output, result, atol=1e-04)
    assert output.shape == (5, 1)


@pytest.mark.torch
def test_position_wise_feed_forward_dropout_at_input():
    """Test invoking PositionwiseFeedForward."""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    layer = torch_layers.PositionwiseFeedForward(d_input=2,
                                                 d_hidden=2,
                                                 d_output=2,
                                                 activation='relu',
                                                 n_layers=1,
                                                 dropout_p=0.0,
                                                 dropout_at_input_no_act=True)
    result = layer(input_ar)
    output_ar = torch.tensor([[0.4810, -1.4331], [1.9771, -5.8426]])
    assert torch.allclose(result, output_ar, rtol=1e-4)


@pytest.mark.torch
def test_sub_layer_connection():
    """Test invoking SublayerConnection."""
    torch.manual_seed(0)
    input_ar = torch.tensor([[1., 2.], [5., 6.]])
    layer = torch_layers.SublayerConnection(2, 0.0)
    result = layer(input_ar, input_ar)
    output_ar = torch.tensor([[2.0027e-05, 3.0000e+00],
                              [4.0000e+00, 7.0000e+00]])
    assert torch.allclose(result, output_ar)


@pytest.mark.torch
def test_mat_encoder_layer():
    """Test invoking MATEncoderLayer."""
    input_smile = "CC"
    feat = dc.feat.MATFeaturizer()
    input_smile = "CC"
    out = feat.featurize(input_smile)
    node = torch.tensor(out[0].node_features).float().unsqueeze(0)
    adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
    dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
    mask = torch.sum(torch.abs(node), dim=-1) != 0
    layer = torch_layers.MATEncoderLayer()
    op = torch_layers.MATEmbedding()(node)
    output = layer(op, mask, adj, dist)
    assert (output.shape == (1, 3, 1024))


@pytest.mark.torch
def test_mat_embedding():
    """Test invoking MATEmbedding."""
    torch.manual_seed(0)
    input_ar = torch.tensor([1., 2., 3.])
    layer = torch_layers.MATEmbedding(3, 1, 0.0)
    result = layer(input_ar).detach()
    output_ar = torch.tensor([-1.2353])
    assert torch.allclose(result, output_ar, rtol=1e-4)


@pytest.mark.torch
def test_mat_generator():
    """Test invoking MATGenerator."""
    torch.manual_seed(0)
    input_ar = torch.tensor([1., 2., 3.])
    layer = torch_layers.MATGenerator(3, 'mean', 1, 1, 0.0)
    mask = torch.tensor([1., 1., 1.])
    result = layer(input_ar, mask)
    output_ar = torch.tensor([-1.4436])
    assert torch.allclose(result, output_ar, rtol=1e-4)


@pytest.mark.torch
def test_dmpnn_encoder_layer():
    """Test invoking DMPNNEncoderLayer."""
    torch.manual_seed(0)

    input_smile = "CC"
    feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
    graph = feat.featurize(input_smile)

    from deepchem.models.torch_models.dmpnn import _MapperDMPNN
    mapper = _MapperDMPNN(graph[0])
    atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values
    molecules_unbatch_key = [len(atom_features)]

    atom_features = torch.from_numpy(atom_features).float()
    f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
    atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
    mapping = torch.from_numpy(mapping)
    global_features = torch.from_numpy(global_features).float()

    layer = torch_layers.DMPNNEncoderLayer(d_hidden=2)
    assert layer.W_i.__repr__(
    ) == 'Linear(in_features=147, out_features=2, bias=False)'
    assert layer.W_h.__repr__(
    ) == 'Linear(in_features=2, out_features=2, bias=False)'
    assert layer.W_o.__repr__(
    ) == 'Linear(in_features=135, out_features=2, bias=True)'

    output = layer(atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds,
                   mapping, global_features, molecules_unbatch_key)
    readout_output = torch.tensor([[0.1116, 0.0470]])
    assert output.shape == torch.Size([1, 2 + 2048])
    assert torch.allclose(output[0][:2], readout_output, atol=1e-4)


@pytest.mark.torch
def test_torch_interatomic_l2_distances():
    """Test Invoking the torch equivalent of InteratomicL2Distances"""
    atoms = 5
    neighbors = 2
    coords = np.random.rand(atoms, 3)
    neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
    layer = torch_layers.InteratomicL2Distances(atoms, neighbors, 3)
    result = layer([coords, neighbor_list])
    assert result.shape == (atoms, neighbors)
    for atom in range(atoms):
        for neighbor in range(neighbors):
            delta = coords[atom] - coords[neighbor_list[atom, neighbor]]
            dist2 = np.dot(delta, delta)
            assert np.allclose(dist2, result[atom, neighbor])


@pytest.mark.torch
def test_torch_neighbor_list():
    """Test invoking the Torch equivalent of NeighborList."""
    N_atoms = 5
    start = 0
    stop = 12
    nbr_cutoff = 3
    ndim = 3
    M_nbrs = 2
    coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
    coords = torch.as_tensor(coords, dtype=torch.float)
    layer = torch_layers.NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                      stop)
    result = layer(coords)
    assert result.shape == (N_atoms, M_nbrs)


@pytest.mark.torch
def test_torch_lstm_step():
    """Test invoking LSTMStep."""
    n_test = 5
    n_feat = 10
    y = np.random.rand(n_test, 2 * n_feat).astype(np.float32)
    state_zero = np.random.rand(n_test, n_feat).astype(np.float32)
    state_one = np.random.rand(n_test, n_feat).astype(np.float32)
    layer = torch_layers.LSTMStep(n_feat, 2 * n_feat)
    result = layer([y, state_zero, state_one])
    h_out, h_copy_out, c_out = (result[0], result[1][0], result[1][1])
    assert h_out.shape == (n_test, n_feat)
    assert h_copy_out.shape == (n_test, n_feat)
    assert c_out.shape == (n_test, n_feat)


@pytest.mark.torch
def test_torch_gru():
    n_hidden = 100
    batch_size = 10
    x = torch.tensor(np.random.rand(batch_size, n_hidden).astype(np.float32))
    h_0 = torch.tensor(np.random.rand(batch_size, n_hidden).astype(np.float32))
    init = 'xavier_uniform_'
    layer = torch_layers.GatedRecurrentUnit(n_hidden, init)
    y = layer([x, h_0])
    assert y.shape == (batch_size, n_hidden)


@pytest.mark.torch
def test_torch_atomic_convolution():
    """Test invoking the Torch equivalent of AtomicConvolution"""
    batch_size = 4
    max_atoms = 5
    max_neighbors = 2
    dimensions = 3
    radial_params = torch.tensor([[5.0, 2.0, 0.5], [10.0, 2.0, 0.5],
                                  [5.0, 1.0, 0.2]])
    input1 = np.random.rand(batch_size, max_atoms,
                            dimensions).astype(np.float32)
    input2 = np.random.randint(max_atoms,
                               size=(batch_size, max_atoms, max_neighbors))
    input3 = np.random.randint(1,
                               10,
                               size=(batch_size, max_atoms, max_neighbors))

    layer = torch_layers.AtomicConvolution(radial_params=radial_params)
    result = layer([input1, input2, input3])
    assert result.shape == (batch_size, max_atoms, len(radial_params))

    atom_types = [1, 2, 8]
    layer = torch_layers.AtomicConvolution(radial_params=radial_params,
                                           atom_types=atom_types)
    result = layer([input1, input2, input3])
    assert result.shape == (batch_size, max_atoms,
                            len(radial_params) * len(atom_types))

    # By setting the `box_size` to effectively zero, the result should only contain `nan`.
    box_size = [0.0, 0.0, 0.0]
    layer = torch_layers.AtomicConvolution(radial_params=radial_params,
                                           box_size=box_size)
    result = layer([input1, input2, input3])
    assert torch.all(result.isnan())

    # Check that layer has three trainable parameters.
    assert len(list(layer.parameters())) == 3

    with pytest.raises(ValueError):
        # Check when `box_size` is of wrong dimensionality.
        dimensions = 2
        box_size = torch.tensor([1.0, 1.0, 1.0])
        input1 = np.random.rand(batch_size, max_atoms,
                                dimensions).astype(np.float32)

        layer = torch_layers.AtomicConvolution(radial_params=radial_params,
                                               box_size=box_size)
        _ = layer([input1, input2, input3])

        # Check when `inputs` is of wrong length.
        layer = torch_layers.AtomicConvolution(radial_params=radial_params)
        _ = layer([input1, input2])


@pytest.mark.torch
def test_torch_combine_mean_std():
    """Test invoking the Torch equivalent of CombineMeanStd."""
    mean = torch.rand((5, 3), dtype=torch.float32)
    std = torch.rand((5, 3), dtype=torch.float32)
    layer = torch_layers.CombineMeanStd(training_only=True, noise_epsilon=0.01)
    result1 = layer([mean, std], training=False)
    assert np.array_equal(result1, mean)  # No noise in test mode
    result2 = layer([mean, std], training=True)
    assert not np.array_equal(result2, mean)
    assert np.allclose(result2, mean, atol=0.1)
    assert result1.shape == mean.shape and result1.shape == std.shape
    assert result2.shape == mean.shape and result2.shape == std.shape


@pytest.mark.torch
def test_torch_weighted_linear_combo():
    """Test invoking the Torch equivalent of WeightedLinearCombo."""
    input1 = np.random.rand(5, 10).astype(np.float32)
    input2 = np.random.rand(5, 10).astype(np.float32)
    layer = torch_layers.WeightedLinearCombo(len([input1, input2]))
    result = layer([input1, input2])
    assert len(layer.input_weights) == 2
    expected = torch.Tensor(input1) * layer.input_weights[0] + torch.Tensor(
        input2) * layer.input_weights[1]
    assert torch.allclose(result, expected)


@pytest.mark.torch
def test_local_global_discriminator():
    import torch
    from deepchem.models.torch_models.gnn import LocalGlobalDiscriminator

    hidden_dim = 10
    discriminator = LocalGlobalDiscriminator(hidden_dim=hidden_dim)

    # Create random local node representations and global graph representations
    batch_size = 6
    x = torch.randn(batch_size, hidden_dim)
    summary = torch.randn(batch_size, hidden_dim)

    # Compute similarity scores using the discriminator
    similarity_scores = discriminator(x, summary)

    # Check if the output has the correct shape and dtype
    assert similarity_scores.shape == (batch_size,)
    assert similarity_scores.dtype == torch.float32


@pytest.mark.torch
def test_set_gather():
    """Test invoking the Torch Equivalent of SetGather."""
    # total_n_atoms = 4
    # n_atom_feat = 4
    # atom_feat = np.random.rand(total_n_atoms, n_atom_feat)
    atom_feat = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "atom_feat_SetGather.npy"))
    atom_split = np.array([0, 0, 1, 1], dtype=np.int32)
    torch_layer = torch_layers.SetGather(2, 2, 4)
    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "weights_SetGather_tf.npy"))
    torch_layer.U = torch.nn.Parameter(torch.from_numpy(weights))
    torch_result = torch_layer([atom_feat, atom_split])
    tf_result = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "result_SetGather_tf.npy"))
    assert np.allclose(np.array(tf_result), np.array(torch_result), atol=1e-4)


@pytest.mark.torch
def test_dtnn_embedding():
    """Test invoking the Torch Equivalent of DTNNEmbedding."""
    # Embeddings and results from Tensorflow implementation
    embeddings_tf = [
        [0.51979446, -0.43430394, -0.73670053, -0.443037, 0.6706989],
        [0.21077824, -0.62696636, 0.66158307, -0.25795913, 0.31941652],
        [-0.26653743, 0.15180665, 0.21961051, -0.7263894, -0.4521287],
        [0.64324486, -0.66274744, 0.2814387, 0.5478991, -0.32046735],
        [0.1925143, -0.5505201, -0.35381562, -0.7409675, 0.6427947]
    ]
    results_tf = [[0.64324486, -0.66274744, 0.2814387, 0.5478991, -0.32046735],
                  [-0.26653743, 0.15180665, 0.21961051, -0.7263894, -0.4521287],
                  [0.1925143, -0.5505201, -0.35381562, -0.7409675, 0.6427947]]
    embedding_layer_torch = torch_layers.DTNNEmbedding(5, 5, 'xavier_uniform_')
    embedding_layer_torch.embedding_list = torch.nn.Parameter(
        torch.tensor(embeddings_tf))
    result_torch = embedding_layer_torch(torch.tensor([3, 2, 4]))
    assert torch.allclose(torch.tensor(results_tf), result_torch)
    assert result_torch.shape == (3, 5)


@pytest.mark.torch
def test_dtnn_step():
    """Test invoking the Torch Equivalent of DTNNEmbedding."""
    # Weights and Embeddings from Tensorflow implementation
    emb = [[-0.57997036, -0.54435134], [0.38634658, -0.7800591],
           [0.48274183, 0.09290886], [0.72841835, -0.21489048]]
    W_cf = [[
        -0.6980064, -0.40244102, 0.12015277, -0.11236137, 0.44983745,
        -0.7261406, 0.03590739, 0.18886101
    ],
            [
                0.38193417, 0.08161169, -0.19805211, 0.01473492, -0.21253234,
                0.07730722, -0.25919884, -0.4723375
            ]]
    W_fc = [[0.27500582, 0.19958842], [-0.07512283, -0.4402059],
            [-0.6734804, -0.13714153], [-0.7683939, 0.04202372],
            [0.61084986, 0.4715314], [0.3767004, -0.59029776],
            [-0.1084643, 0.34647202], [-0.656258, -0.3710086]]
    W_df = [[
        -0.53450876, -0.49899083, 0.27615517, -0.15492862, 0.61964273,
        0.18540198, 0.17524064, -0.3806646
    ],
            [
                0.24792421, -0.38151026, -0.50989795, -0.16949275, -0.1911948,
                0.24427831, 0.3103531, -0.548931
            ],
            [
                0.5648807, -0.26876533, -0.4311456, 0.03692579, -0.04565948,
                0.6494999, -0.489844, -0.6053973
            ],
            [
                -0.5715633, 0.5406003, -0.4798649, -0.6116994, 0.1802761,
                -0.02659523, -0.14560652, -0.59008956
            ],
            [
                -0.64630675, -0.2756685, -0.43883026, 0.14410889, -0.13292378,
                -0.17106324, -0.60326487, -0.25875738
            ],
            [
                -0.28023764, 0.54396844, -0.05222553, -0.6502703, -0.5865139,
                -0.03999609, -0.16664535, 0.5127555
            ]]
    output_tf = [[[-0.5800, -0.5444], [0.3863, -0.7801], [0.4827, 0.0929],
                  [0.7284, -0.2149]],
                 [[0.2256, -0.1123], [1.1920, -0.3480], [1.2884, 0.5249],
                  [1.5340, 0.2171]]]
    step_layer = torch_layers.DTNNStep(4, 6, 8)
    step_layer.W_fc = torch.nn.Parameter(torch.Tensor(W_fc))
    step_layer.W_cf = torch.nn.Parameter(torch.Tensor(W_cf))
    step_layer.W_df = torch.nn.Parameter(torch.Tensor(W_df))
    output_torch = step_layer([
        torch.Tensor(emb),
        torch.Tensor([0, 1, 2, 3, 4, 5]).to(torch.float32),
        torch.Tensor([1]).to(torch.int64),
        torch.Tensor([[1]]).to(torch.int64)
    ])
    assert torch.allclose(torch.tensor(output_tf), output_torch, atol=1e-4)
    assert output_torch.shape == (2, 4, 2)


@pytest.mark.torch
def test_dtnn_gather():
    """Test invoking the Torch equivalent of EdgeNetwork."""
    W_list_1 = [[
        0.54732025, -0.627077, -0.2903021, -0.53665423, -0.00559229,
        -0.32349566, 0.1962483, 0.5581455, 0.11647487, 0.13117266
    ],
                [
                    -0.66846573, -0.28275022, 0.06701428, 0.43692493,
                    -0.24846172, 0.41073883, -0.04701298, -0.23764172,
                    -0.16597754, -0.23689681
                ],
                [
                    -0.41830233, -0.2093746, 0.11161888, -0.61909866,
                    -0.07230109, 0.20211416, 0.07490742, -0.52804005,
                    -0.4896497, 0.63919294
                ]]
    W_list_2 = [[-0.33358562, -0.5884317, 0.26542962],
                [-0.6087704, -0.5719125, -0.05134851],
                [0.19017327, -0.5240722, 0.28907597],
                [0.09558785, 0.2324171, 0.395795],
                [0.04189491, -0.2537845, 0.1019693],
                [-0.27015388, -0.53264153, 0.04725528],
                [-0.03956562, 0.678604, 0.37642324],
                [0.3477502, 0.48643565, -0.48160803],
                [0.29909176, -0.4186227, 0.53793466],
                [0.05536985, 0.64485407, 0.5148499]]
    result_tf = [[0.45788735, 0.9619317, -0.53767115]]

    gather_layer_torch = torch_layers.DTNNGather(3, 3, [10])
    gather_layer_torch.W_list = torch.nn.ParameterList()
    gather_layer_torch.W_list.append(torch.tensor(W_list_1))
    gather_layer_torch.W_list.append(torch.tensor(W_list_2))
    result_torch = gather_layer_torch([
        torch.Tensor([[3, 2, 1]]).to(torch.float32),
        torch.Tensor([0]).to(torch.int64)
    ])

    assert torch.allclose(result_torch, torch.tensor(result_tf), atol=1e-4)
    assert result_torch.shape == (1, 3)


@pytest.mark.torch
def test_edge_network():
    """Test invoking the Torch equivalent of EdgeNetwork."""
    # init parameters
    n_pair_features = 14
    n_hidden = 75  # based on weave featurizer
    torch_init = 'xavier_uniform_'

    # generate features for testing
    mols = ["CCC"]
    featurizer = dc.feat.WeaveFeaturizer()
    features = featurizer.featurize(mols)
    X_b = np.asarray([features[0]])
    X_b = dc.data.pad_features(1, X_b)

    atom_feat = []
    pair_feat = []
    atom_to_pair = []
    start = 0
    for mol in X_b:
        n_atoms = mol.get_num_atoms()

        # index of pair features
        C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
        atom_to_pair.append(
            np.transpose(np.array([C1.flatten() + start,
                                   C0.flatten() + start])))
        start = start + n_atoms

        # atom features
        atom_feat.append(mol.get_atom_features())

        # pair features
        pair_feat.append(
            np.reshape(mol.get_pair_features(),
                       (n_atoms * n_atoms, n_pair_features)))

    atom_features = np.concatenate(atom_feat, axis=0)
    pair_features = np.concatenate(pair_feat, axis=0)
    atom_to_pair_array = np.concatenate(atom_to_pair, axis=0)

    # tensors for torch layer
    torch_pair_features = torch.Tensor(pair_features)
    torch_atom_features = torch.Tensor(atom_features)
    torch_atom_to_pair = torch.Tensor(atom_to_pair_array)
    torch_atom_to_pair = torch.squeeze(torch_atom_to_pair.to(torch.int64),
                                       dim=0)

    torch_inputs = [
        torch_pair_features, torch_atom_features, torch_atom_to_pair
    ]

    torch_layer = dc.models.torch_models.layers.EdgeNetwork(
        n_pair_features, n_hidden, torch_init)

    # assigning tensorflow layer weights to torch layer
    torch_layer.W = torch.from_numpy(
        np.load("deepchem/models/tests/assets/edgenetwork_weights.npy"))

    torch_result = torch_layer(torch_inputs)

    assert np.allclose(
        np.array(torch_result),
        np.load("deepchem/models/tests/assets/edgenetwork_result.npy"),
        atol=1e-04)


@pytest.mark.torch
def test_mxmnet_envelope():
    """Test for _MXMNetEnvelope helper layer."""
    env = dc.models.torch_models.layers._MXMNetEnvelope(exponent=2)
    input_tensor = torch.tensor([0.5, 1.0, 2.0, 3.0])
    output = env(input_tensor)
    output = output.detach().numpy()
    result = np.array([1.3125, 0.0000, 0.0000, 0.0000])
    assert np.allclose(result, output, atol=1e-04)


@pytest.mark.torch
def test_mxmnet_global_message_passing():
    """ Test for MXMNetGlobalMessagePassing Layer."""
    seed = 123
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dim = 1
    node_features = torch.tensor([[0.8343], [1.2713], [1.2713], [1.2713],
                                  [1.2713]])

    edge_attr = torch.tensor([[1.0004], [1.0004], [1.0005], [1.0004], [1.0004],
                              [-0.2644], [-0.2644], [-0.2644], [1.0004],
                              [-0.2644], [-0.2644], [-0.2644], [1.0005],
                              [-0.2644], [-0.2644], [-0.2644], [1.0004],
                              [-0.2644], [-0.2644], [-0.2644]])

    edge_indices = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
         [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])

    out = dc.models.torch_models.layers.MXMNetGlobalMessagePassing(dim)
    output = out(node_features, edge_attr, edge_indices)
    output = output.detach().numpy()
    result = np.array([[-0.27947044], [2.417905], [2.417905], [2.4178727],
                       [2.417905]])
    print(output)
    assert np.allclose(output, result, atol=1e-04)
    assert output.shape == (5, 1)


@pytest.mark.torch
def test_mxmnet_besselbasis():
    """Test for MXMNetBesselBasisLayer"""

    radial_layer = dc.models.torch_models.layers.MXMNetBesselBasisLayer(
        num_radial=2, cutoff=2.0, envelope_exponent=2)
    distances = torch.tensor([0.5, 1.0, 2.0, 3.0])
    output = radial_layer(distances)
    output = output.detach().numpy()
    result = np.array([[2.6434e+00, 3.7383e+00], [1.3125e+00, -1.1474e-07],
                       [-0.0000e+00, 0.0000e+00], [-0.0000e+00, -0.0000e+00]])
    assert np.allclose(result, output, atol=1e-04)


@pytest.mark.torch
def test_variational_randomiser():
    """Test invoking the Torch equivalent of VariationalRandomizer"""
    dense_mean_W = np.array(
        [[0.14599681, -0.57214814, -0.09179449, -0.56948453, -0.00163919],
         [-0.00663584, 0.08912718, -0.5397246, 0.62155735, -0.55921054],
         [-0.40077654, -0.7022078, -0.65213305, 0.13691288, 0.21464229],
         [-0.31985036, 0.30251813, -0.28222048, -0.7732454, -0.58668435],
         [0.13017195, 0.5091233, 0.6633384, -0.6581168, 0.31816483]])
    dense_stddev_W = np.array(
        [[-0.59771556, 0.18744493, -0.1720407, -0.48102698, 0.15075064],
         [-0.3724205, 0.55220056, 0.739933, 0.2427752, -0.00584781],
         [-0.6358649, 0.631323, 0.6159171, 0.5158607, 0.28268492],
         [0.54178894, 0.20224595, 0.04413438, -0.7469158, 0.28610182],
         [-0.332127, 0.35822368, -0.28074694, 0.16514707, -0.768287]])
    embeddings = np.array(
        [[[-1.7328764, -1.0216347, -0.03802864, 0.4724247, 0.45277336],
          [-0.64516085, 1.2509763, 1.1879151, -0.36354527, 2.1101253],
          [-0.05635107, 0.7441244, -0.29123098, 0.29690874, 1.6139926]],
         [[-1.7231425, -0.09280606, 1.0805027, -0.3756243, 0.50924677],
          [-0.66316897, 0.5539947, -0.32553753, -0.61652315, -0.50795805],
          [0.9577992, 0.4545413, 1.883168, -0.10543215, -0.7342148]]])
    tf_output = np.array(
        [[[-0.32314086, 1.3005452, 0.90228367, -0.31664285, 0.43287927],
          [-0.18762197, 0.6107953, 0.11168798, 0.20000434, 0.44112915],
          [0.21868376, 1.2146091, 0.7803014, -0.83704513, -0.13921635]],
         [[-0.4975644, 0.36452, -0.05255505, 1.0268594, 0.6690416],
          [0.16104428, 0.2122792, -0.18878815, 1.4884531, -0.17849664],
          [-0.6797619, -2.2355673, -2.0186017, 0.5596257, -0.02329272]]])

    embedding_dimension = 5
    annealing_start_step = 10
    annealing_final_step = 20
    embedding_shape = embeddings.shape
    global_step = torch.tensor(range(5), dtype=torch.int32)
    layer = torch_layers.VariationalRandomizer(embedding_dimension,
                                               annealing_start_step,
                                               annealing_final_step)

    layer.dense_mean.weight = torch.nn.Parameter(
        torch.tensor(dense_mean_W.transpose(), dtype=torch.float32))
    layer.dense_stddev.weight = torch.nn.Parameter(
        torch.tensor(dense_stddev_W.transpose(), dtype=torch.float32))
    output = layer([torch.tensor(embeddings, dtype=torch.float32), global_step],
                   False)

    assert output.shape == embedding_shape
    assert np.allclose(output.detach().numpy(), tf_output, atol=1e-4)


@pytest.mark.torch
def test_encoder_rnn():
    """Test for Encoder Layer of SeqToSeq Model"""
    hidden_size = 7
    num_input_token = 4
    n_layers = 5
    input = torch.tensor([[1, 0, 2, 3, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    layer = torch_layers.EncoderRNN(num_input_token, hidden_size, n_layers)
    emb, hidden = layer(input)

    assert emb.shape == emb.shape == (input.shape[0], input.shape[1],
                                      hidden_size)
    assert hidden.shape == (input.shape[0], hidden_size)


@pytest.mark.torch
def test_decoder_rnn():
    """Test for Decoder Layer of SeqToSeq Model"""
    embedding_dimensions = 5
    num_output_tokens = 7
    max_length = 4
    batch_size = 2
    n_layers = 9
    layer = torch_layers.DecoderRNN(embedding_dimensions, num_output_tokens,
                                    n_layers, max_length, batch_size)
    embeddings = torch.randn(batch_size, embedding_dimensions)
    output, hidden = layer([embeddings, None])
    assert output.shape == (batch_size, max_length, num_output_tokens)
    assert hidden.shape == (n_layers, batch_size, embedding_dimensions)


@pytest.mark.torch
def test_FerminetElectronFeature():
    "Test for FerminetElectronFeature layer."
    electron_layer = dc.models.torch_models.layers.FerminetElectronFeature(
        [32, 32, 32], [16, 16, 16], 4, 8, 10, [5, 5])
    one_electron_test = torch.randn(8, 10, 4 * 4)
    two_electron_test = torch.randn(8, 10, 10, 4)
    one, two = electron_layer.forward(one_electron_test, two_electron_test)
    assert one.size() == torch.Size([8, 10, 32])
    assert two.size() == torch.Size([8, 10, 10, 16])


@pytest.mark.torch
def test_FerminetEnvelope():
    "Test for FerminetEnvelope layer."
    envelope_layer = dc.models.torch_models.layers.FerminetEnvelope(
        [32, 32, 32], [16, 16, 16], 10, 8, [5, 5], 5, 16)
    one_electron = torch.randn(8, 10, 32)
    one_electron_permuted = torch.randn(8, 10, 5, 3)
    psi, _, _ = envelope_layer.forward(one_electron, one_electron_permuted)
    assert psi.size() == torch.Size([8])


@pytest.mark.torch
def test_mxmnet_local_message_passing():
    """ Test for MXMNetLocalMessagePassing Layer."""
    seed = 123
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dim = 1
    h = torch.tensor([[0.8343], [1.2713], [1.2713], [1.2713], [1.2713]])

    rbf = torch.tensor([[-0.2628], [-0.2628], [-0.2628], [-0.2628], [-0.2629],
                        [-0.2629], [-0.2628], [-0.2628]])

    sbf1 = torch.tensor([[-0.2767], [-0.2767], [-0.2767], [-0.2767], [-0.2767],
                         [-0.2767], [-0.2767], [-0.2767], [-0.2767], [-0.2767],
                         [-0.2767], [-0.2767]])

    sbf2 = torch.tensor([[-0.0301], [-0.0301], [-0.1483], [-0.1486], [-0.1484],
                         [-0.0301], [-0.1483], [-0.0301], [-0.1485], [-0.1483],
                         [-0.0301], [-0.1486], [-0.1485], [-0.0301], [-0.1486],
                         [-0.0301], [-0.1484], [-0.1483], [-0.1486], [-0.0301]])

    idx_kj = torch.tensor([3, 5, 7, 1, 5, 7, 1, 3, 7, 1, 3, 5])

    idx_ji_1 = torch.tensor([0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6])

    idx_jj = torch.tensor(
        [0, 1, 3, 5, 7, 2, 1, 3, 5, 7, 4, 1, 3, 5, 7, 6, 1, 3, 5, 7])

    idx_ji_2 = torch.tensor(
        [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 7, 7, 7, 7])

    edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4],
                               [1, 0, 2, 0, 3, 0, 4, 0]])

    layer = dc.models.torch_models.layers.MXMNetLocalMessagePassing(
        dim, activation_fn='silu')

    output = layer(h, rbf, sbf1, sbf2, idx_kj, idx_ji_1, idx_jj, idx_ji_2,
                   edge_index)
    result0 = np.array([[0.7916], [1.2796], [1.2796], [1.2796], [1.2796]])
    result1 = np.array([[0.3439], [0.3441], [0.3441], [0.3441], [0.3441]])

    assert np.allclose(result0, output[0].detach().numpy(), atol=1e-04)
    assert np.allclose(result1, output[1].detach().numpy(), atol=1e-04)

    assert output[0].shape == (5, 1)
    assert output[1].shape == (5, 1)


@pytest.mark.torch
def test_mxmnet_sphericalbasis():
    """Test for MXMNetSphericalBasisLayer"""

    dist = torch.tensor([0.5, 1.0, 2.0, 3.0])
    angle = torch.tensor([0.1, 0.2, 0.3, 0.4])
    idx_kj = torch.tensor([0, 1, 2, 3])
    spherical_layer = dc.models.torch_models.layers.MXMNetSphericalBasisLayer(
        envelope_exponent=2, num_spherical=2, num_radial=2, cutoff=2.0)
    output = spherical_layer(dist, angle, idx_kj)
    output = output.detach().numpy()

    result = np.array([[4.2182e+00, 5.9654e+00, 3.8959e+00, 8.6795e+00],
                       [1.0472e+00, -9.1551e-08, 1.7717e+00, 1.0400e+00],
                       [-0.0000e+00, 0.0000e+00, -0.0000e+00, -0.0000e+00],
                       [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00]])

    assert np.allclose(output, result, atol=1e-04)


@pytest.mark.torch
def test_torch_highway_layer():
    """Test invoking HighwayLayer."""
    batch_size = 10
    feat_dim = 5
    input_tensor = torch.randn(batch_size, feat_dim)
    highway_layer = torch_layers.HighwayLayer(d_input=input_tensor.shape[1])
    output_tensor = highway_layer(input_tensor)

    assert output_tensor.shape == (batch_size, feat_dim)


@pytest.mark.torch
def test_torch_graph_conv():
    """Test invoking GraphConv."""
    out_channels = 2
    n_atoms = 4  # In CCC and C, there are 4 atoms
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
    args = [atom_features, degree_slice, membership] + deg_adjs
    layer = torch_layers.GraphConv(
        out_channels, number_input_features=atom_features.shape[-1])
    torch.set_printoptions(precision=8)
    W_list = np.load("deepchem/models/tests/assets/graphconvlayer_weights.npy",
                     allow_pickle=True).tolist()
    layer.W_list = nn.ParameterList(
        [nn.Parameter(torch.tensor(k)) for k in W_list])
    b_list = np.load("deepchem/models/tests/assets/graphconvlayer_biases.npy",
                     allow_pickle=True).tolist()
    layer.b_list = nn.ParameterList(
        [nn.Parameter(torch.tensor(k)) for k in b_list])
    result = layer(args)
    assert np.allclose(
        result.detach().numpy(),
        np.load("deepchem/models/tests/assets/graphconvlayer_result.npy"),
        atol=1e-4)
    assert result.shape == (n_atoms, out_channels)
    num_deg = 2 * layer.max_degree + (1 - layer.min_degree)
    assert len(list(layer.parameters())) == 2 * num_deg


@pytest.mark.torch
def test_torch_graph_pool():
    """Test invoking GraphPool."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
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
    args = [atom_features, degree_slice, membership] + deg_adjs
    result = torch_layers.GraphPool()(args)
    assert np.allclose(
        result.detach().numpy(),
        np.load("deepchem/models/tests/assets/graphpoollayer_result.npy"),
        atol=1e-4)
    assert result.shape[0] == n_atoms


@pytest.mark.torch
def test_torch_graph_gather():
    import deepchem as dc
    batch_size = 2
    n_features = 75
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
    args = [atom_features, degree_slice, membership] + deg_adjs
    result = torch_layers.GraphGather(batch_size)(args)
    assert np.allclose(
        result.detach().numpy(),
        np.load("deepchem/models/tests/assets/graphgatherlayer_result.npy"),
        atol=1e-4)
    assert result.shape == (batch_size, 2 * n_features)


@pytest.mark.torch
def test_torch_cosine_dist():
    """Test invoking cosine_dist."""

    x = torch.ones((5, 4), dtype=torch.float32)
    y_same = torch.ones((5, 4), dtype=torch.float32)
    # x and y are the same tensor (equivalent at every element)
    # the pairwise inner product of the rows in x and y will always be 1
    # the output tensor will be of shape (5,5)
    cos_sim_same = torch_layers.cosine_dist(x, y_same)
    diff = cos_sim_same - torch.ones((5, 5), dtype=torch.float32)
    assert torch.abs(torch.sum(diff)) < 1e-5  # True

    identity_tensor = torch.eye(
        16, dtype=torch.float32)  # identity matrix of shape (16,16)
    x1 = identity_tensor[0:8, :]  # first half of the identity matrix
    x2 = identity_tensor[8:16, :]  # second half of the identity matrix
    # each row in x1 is orthogonal to each row in x2
    # the pairwise inner product of the rows in x1 and x2 will always be 0
    # the output tensor will be of shape (8,8)
    cos_sim_orth = torch_layers.cosine_dist(x1, x2)
    assert torch.abs(torch.sum(cos_sim_orth)) < 1e-5  # True
    assert all([cos_sim_orth.shape[dim] == 8 for dim in range(2)])  # True

    x3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    x4 = torch.tensor([[2.0, 1.0], [0.0, -1.0]], dtype=torch.float32)
    pre_calculated_value = torch.tensor([[0.79999995, -0.8944272],
                                         [0.8944272, -0.8]])
    result = torch_layers.cosine_dist(x3, x4)
    assert torch.allclose(result, pre_calculated_value, atol=1e-4)
