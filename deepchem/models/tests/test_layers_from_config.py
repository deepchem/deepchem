import deepchem as dc
import pytest
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.python.eager import context  # noqa: F401
    has_tensorflow = True
except:
    has_tensorflow = False


@pytest.mark.tensorflow
def test_interatomic_l2_distance():
    N_atoms = 10
    M_nbrs = 15
    ndim = 20

    layer = dc.models.layers.InteratomicL2Distances(N_atoms=N_atoms,
                                                    M_nbrs=M_nbrs,
                                                    ndim=ndim)
    config = layer.get_config()
    layer_copied = dc.models.layers.InteratomicL2Distances.from_config(config)

    assert layer_copied.N_atoms == layer.N_atoms
    assert layer_copied.M_nbrs == layer.M_nbrs
    assert layer_copied.ndim == layer.ndim


@pytest.mark.tensorflow
def test_graph_conv():
    out_channel = 10
    min_deg = 0,
    max_deg = 10,
    activation_fn = 'relu'

    layer = dc.models.layers.GraphConv(out_channel=out_channel,
                                       min_deg=min_deg,
                                       max_deg=max_deg,
                                       activation_fn=activation_fn)
    config = layer.get_config()
    layer_copied = dc.models.layers.GraphConv.from_config(config)

    assert layer_copied.out_channel == layer.out_channel
    assert layer_copied.activation_fn == layer.activation_fn
    assert layer_copied.max_degree == layer.max_degree
    assert layer_copied.min_degree == layer.min_degree


@pytest.mark.tensorflow
def test_graph_gather():
    batch_size = 10
    activation_fn = 'relu'

    layer_copied = dc.models.layers.GraphGather(batch_size=batch_size,
                                                activation_fn=activation_fn)
    config = layer_copied.get_config()
    layer_copied = dc.models.layers.GraphGather.from_config(config)

    assert layer_copied.batch_size == layer_copied.batch_size
    assert layer_copied.activation_fn == layer_copied.activation_fn


@pytest.mark.tensorflow
def test_graph_pool():
    min_degree = 0
    max_degree = 10

    layer_copied = dc.models.layers.GraphPool(min_degree=min_degree,
                                              max_degree=max_degree)
    config = layer_copied.get_config()
    layer_copied = dc.models.layers.GraphPool.from_config(config)

    assert layer_copied.max_degree == layer_copied.max_degree
    assert layer_copied.min_degree == layer_copied.min_degree


@pytest.mark.tensorflow
def test_lstmstep():
    output_dim = 100
    input_dim = 50
    init_fn = 'glorot_uniform'
    inner_init_fn = 'orthogonal'
    activation_fn = 'tanh'
    inner_activation_fn = 'hard_sigmoid'

    layer = dc.models.layers.LSTMStep(output_dim, input_dim, init_fn,
                                      inner_init_fn, activation_fn,
                                      inner_activation_fn)
    config = layer.get_config()
    layer_copied = dc.models.layers.LSTMStep.from_config(config)

    assert layer_copied.output_dim == layer.output_dim
    assert layer_copied.input_dim == layer.input_dim
    assert layer_copied.init == layer.init
    assert layer_copied.inner_init == layer.inner_init
    assert layer_copied.activation == layer.activation
    assert layer_copied.inner_activation == layer.inner_activation


@pytest.mark.tensorflow
def test_attn_lstm_embedding():
    n_test = 10
    n_support = 100
    n_feat = 20
    max_depth = 3

    layer = dc.models.layers.AttnLSTMEmbedding(n_test, n_support, n_feat,
                                               max_depth)
    config = layer.get_config()
    layer_copied = dc.models.layers.AttnLSTMEmbedding.from_config(config)

    assert layer_copied.n_test == layer.n_test
    assert layer_copied.n_support == layer.n_support
    assert layer_copied.n_feat == layer.n_feat
    assert layer_copied.max_depth == layer.max_depth


@pytest.mark.tensorflow
def test_iterref_lstm_embedding():
    n_test = 10
    n_support = 100
    n_feat = 20
    max_depth = 3

    layer = dc.models.layers.IterRefLSTMEmbedding(n_test, n_support, n_feat,
                                                  max_depth)
    config = layer.get_config()
    layer_copied = dc.models.layers.IterRefLSTMEmbedding.from_config(config)

    assert layer_copied.n_test == layer.n_test
    assert layer_copied.n_support == layer.n_support
    assert layer_copied.n_feat == layer.n_feat
    assert layer_copied.max_depth == layer.max_depth


@pytest.mark.tensorflow
def test_switched_dropout():
    rate = 0.1
    layer = dc.models.layers.SwitchedDropout(rate=rate)
    config = layer.get_config()
    layer_copied = dc.models.layers.SwitchedDropout.from_config(config)

    assert layer_copied.rate == layer.rate


@pytest.mark.tensorflow
def test_weighted_linearcombo():
    std = 0.1
    layer = dc.models.layers.WeightedLinearCombo(std=std)

    config = layer.get_config()
    layer_copied = dc.models.layers.WeightedLinearCombo.from_config(config)

    assert layer_copied.std == layer.std


@pytest.mark.tensorflow
def test_combine_mean_std():
    training_only = True
    noise_epsilon = 0.001

    layer = dc.models.layers.CombineMeanStd(training_only, noise_epsilon)
    config = layer.get_config()
    layer_copied = dc.models.layers.CombineMeanStd.from_config(config)

    assert layer_copied.training_only == layer.training_only
    assert layer_copied.noise_epsilon == layer.noise_epsilon


@pytest.mark.tensorflow
def test_stack():
    axis = 2
    layer = dc.models.layers.Stack(axis=axis)
    config = layer.get_config()
    layer_copied = dc.models.layers.Stack.from_config(config)

    assert layer_copied.axis == layer.axis


@pytest.mark.tensorflow
def test_variable():
    initial_value = 10
    layer = dc.models.layers.Variable(initial_value)
    config = layer.get_config()
    layer_copied = dc.models.layers.Variable.from_config(config)

    assert layer_copied.initial_value == layer.initial_value


@pytest.mark.tensorflow
def test_vina_free_energy():
    N_atoms = 10
    M_nbrs = 15
    ndim = 20
    nbr_cutoff = 5
    start = 1
    stop = 7
    stddev = 0.3
    Nrot = 1

    layer = dc.models.layers.VinaFreeEnergy(N_atoms, M_nbrs, ndim, nbr_cutoff,
                                            start, stop, stddev, Nrot)
    config = layer.get_config()
    layer_copied = dc.models.layers.VinaFreeEnergy.from_config(config)

    assert layer_copied.N_atoms == layer.N_atoms
    assert layer_copied.M_nbrs == layer.M_nbrs
    assert layer_copied.ndim == layer.ndim
    assert layer_copied.nbr_cutoff == layer.nbr_cutoff
    assert layer_copied.start == layer.start
    assert layer_copied.stop == layer.stop
    assert layer_copied.stddev == layer.stddev
    assert layer_copied.Nrot == layer_copied.Nrot


@pytest.mark.tensorflow
def test_neighbor_list():
    N_atoms = 10
    M_nbrs = 15
    ndim = 20
    nbr_cutoff = 5
    start = 1
    stop = 7

    layer = dc.models.layers.NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff,
                                          start, stop)
    config = layer.get_config()
    layer_copied = dc.models.layers.VinaFreeEnergy.from_config(config)

    assert layer_copied.N_atoms == layer.N_atoms
    assert layer_copied.M_nbrs == layer.M_nbrs
    assert layer_copied.ndim == layer.ndim
    assert layer_copied.nbr_cutoff == layer.nbr_cutoff
    assert layer_copied.start == layer.start
    assert layer_copied.stop == layer.stop


@pytest.mark.tensorflow
def test_atomic_convolution():
    atom_types = None
    radial_params = list()
    boxsize = None

    layer = dc.models.layers.AtomicConvolution(atom_types, radial_params,
                                               boxsize)
    config = layer.get_config()
    layer_copied = dc.models.layers.AtomicConvolution.from_config(config)

    assert layer_copied.atom_types == layer.atom_types
    assert layer_copied.radial_params == layer.radial_params
    assert layer_copied.boxsize == layer.boxsize


@pytest.mark.tensorflow
def test_ani_feat():
    max_atoms = 23
    radial_cutoff = 4.6
    angular_cutoff = 3.1
    radial_length = 32
    angular_length = 8
    atom_cases = [1, 6, 7, 8, 16]
    atomic_number_differentiated = True
    coordinates_in_bohr = True

    layer = dc.models.layers.ANIFeat(max_atoms, radial_cutoff, angular_cutoff,
                                     radial_length, angular_length, atom_cases,
                                     atomic_number_differentiated,
                                     coordinates_in_bohr)
    config = layer.get_config()
    layer_copied = dc.models.layers.ANIFeat.from_config(config)

    assert layer_copied.max_atoms == layer.max_atoms
    assert layer_copied.radial_cutoff == layer.radial_cutoff
    assert layer_copied.angular_cutoff == layer.angular_cutoff
    assert layer_copied.radial_length == layer.radial_length
    assert layer_copied.angular_length == layer.angular_length
    assert layer_copied.atom_cases == layer.atom_cases
    assert layer_copied.atomic_number_differentiated == layer.atomic_number_differentiated
    assert layer_copied.coordinates_in_bohr == layer.coordinates_in_bohr


@pytest.mark.tensorflow
def test_graph_embed_pool():
    num_vertices = 100
    layer = dc.models.layers.GraphEmbedPoolLayer(num_vertices)
    config = layer.get_config()
    layer_copied = dc.models.layers.GraphEmbedPoolLayer.from_config(config)

    assert layer_copied.num_vertices == layer.num_vertices


@pytest.mark.tensorflow
def test_graph_cnn():
    num_filters = 20
    layer = dc.models.layers.GraphCNN(num_filters)
    config = layer.get_config()
    layer_copied = dc.models.layers.GraphCNN.from_config(config)

    assert layer_copied.num_filters == layer.num_filters


@pytest.mark.tensorflow
def test_highway():
    activation_fn = 'relu'
    biases_initializer = 'zeros'
    weights_initializer = None

    layer = dc.models.layers.Highway(activation_fn, biases_initializer,
                                     weights_initializer)
    config = layer.get_config()
    layer_copied = dc.models.layers.Highway.from_config(config)

    assert layer_copied.activation_fn == layer.activation_fn
    assert layer_copied.biases_initializer == layer.biases_initializer
    assert layer_copied.weights_initializer == layer.weights_initializer


@pytest.mark.tensorflow
def test_weave():
    n_atom_input_feat = 75
    n_pair_input_feat = 14
    n_atom_output_feat = 50
    n_pair_output_feat = 50
    n_hidden_AA = 50
    n_hidden_PA = 50
    n_hidden_AP = 50
    n_hidden_PP = 50
    update_pair = True
    init = 'glorot_uniform'
    activation = 'relu'
    batch_normalize = True
    batch_normalize_kwargs = {"renorm": True}

    layer = dc.models.layers.WeaveLayer(n_atom_input_feat, n_pair_input_feat,
                                        n_atom_output_feat, n_pair_output_feat,
                                        n_hidden_AA, n_hidden_PA, n_hidden_AP,
                                        n_hidden_PP, update_pair, init,
                                        activation, batch_normalize,
                                        batch_normalize_kwargs)
    config = layer.get_config()
    layer_copied = dc.models.layers.WeaveLayer.from_config(config)

    assert layer_copied.n_atom_input_feat == layer.n_atom_input_feat
    assert layer_copied.n_pair_input_feat == layer.n_pair_input_feat
    assert layer_copied.n_atom_output_feat == layer.n_atom_output_feat
    assert layer_copied.n_pair_output_feat == layer.n_pair_output_feat
    assert layer_copied.n_hidden_AA == layer.n_hidden_AA
    assert layer_copied.n_hidden_PA == layer.n_hidden_PA
    assert layer_copied.n_hidden_AP == layer.n_hidden_AP
    assert layer_copied.n_hidden_PP == layer.n_hidden_PP
    assert layer_copied.update_pair == layer.update_pair
    assert layer_copied.init == layer.init
    assert layer_copied.activation == layer.activation
    assert layer_copied.batch_normalize == layer.batch_normalize
    assert layer_copied.batch_normalize_kwargs == layer.batch_normalize_kwargs


@pytest.mark.tensorflow
def test_weave_gather():
    batch_size = 32
    n_input = 128
    gaussian_expand = True
    compress_post_gaussian_expansion = False
    init = 'glorot_uniform'
    activation = 'tanh'

    layer = dc.models.layers.WeaveGather(batch_size, n_input, gaussian_expand,
                                         compress_post_gaussian_expansion, init,
                                         activation)
    config = layer.get_config()
    layer_copied = dc.models.layers.WeaveGather.from_config(config)

    assert layer_copied.batch_size == layer.batch_size
    assert layer_copied.n_input == layer.n_input
    assert layer_copied.gaussian_expand == layer.gaussian_expand
    assert layer_copied.compress_post_gaussian_expansion == layer.compress_post_gaussian_expansion
    assert layer_copied.init == layer.init
    assert layer_copied.activation == layer.activation


@pytest.mark.tensorflow
def test_dtnn_embedding():
    n_embedding = 30
    periodic_table_length = 30
    init = 'glorot_uniform'

    layer = dc.models.layers.DTNNEmbedding(n_embedding, periodic_table_length,
                                           init)
    config = layer.get_config()
    layer_copied = dc.models.layers.DTNNEmbedding.from_config(config)

    assert layer_copied.n_embedding == layer.n_embedding
    assert layer_copied.periodic_table_length == layer.periodic_table_length
    assert layer_copied.init == layer.init


@pytest.mark.tensorflow
def test_dtnn_step():
    n_embedding = 30
    n_distance = 100
    n_hidden = 60
    init = 'glorot_uniform'
    activation = 'tanh'

    layer = dc.models.layers.DTNNStep(n_embedding, n_distance, n_hidden, init,
                                      activation)
    config = layer.get_config()
    layer_copied = dc.models.layers.DTNNStep.from_config(config)

    assert layer_copied.n_embedding == layer.n_embedding
    assert layer_copied.n_distance == layer.n_distance
    assert layer_copied.n_hidden == layer.n_hidden
    assert layer_copied.init == layer.init
    assert layer_copied.activation == layer.activation


@pytest.mark.tensorflow
def test_dtnn_gather():
    n_embedding = 30
    n_outputs = 100
    layer_sizes = [100]
    output_activation = True
    init = 'glorot_uniform'
    activation = 'tanh'

    layer = dc.models.layers.DTNNGather(n_embedding, n_outputs, layer_sizes,
                                        output_activation, init, activation)
    config = layer.get_config()
    layer_copied = dc.models.layers.DTNNGather.from_config(config)

    assert layer_copied.n_embedding == layer.n_embedding
    assert layer_copied.n_outputs == layer.n_outputs
    assert layer_copied.layer_sizes == layer.layer_sizes
    assert layer_copied.output_activation == layer.output_activation
    assert layer_copied.init == layer.init
    assert layer_copied.activation == layer.activation


@pytest.mark.tensorflow
def test_dag():
    n_graph_feat = 30
    n_atom_feat = 75
    max_atoms = 50
    layer_sizes = [100]
    init = 'glorot_uniform'
    activation = 'relu'
    dropout = None
    batch_size = 64

    layer = dc.models.layers.DAGLayer(n_graph_feat, n_atom_feat, max_atoms,
                                      layer_sizes, init, activation, dropout,
                                      batch_size)
    config = layer.get_config()
    layer_copied = dc.models.layers.DAGLayer.from_config(config)

    assert layer_copied.n_graph_feat == layer.n_graph_feat
    assert layer_copied.n_atom_feat == layer.n_atom_feat
    assert layer_copied.max_atoms == layer.max_atoms
    assert layer_copied.layer_sizes == layer.layer_sizes
    assert layer_copied.init == layer.init
    assert layer_copied.activation == layer.activation
    assert layer_copied.dropout == layer.dropout
    assert layer_copied.batch_size == layer.batch_size


@pytest.mark.tensorflow
def test_dag_gather():
    n_graph_feat = 30
    n_outputs = 30
    max_atoms = 50
    layer_sizes = [100]
    init = 'glorot_uniform'
    activation = 'relu'
    dropout = None

    layer = dc.models.layers.DAGGather(n_graph_feat, n_outputs, max_atoms,
                                       layer_sizes, init, activation, dropout)
    config = layer.get_config()
    layer_copied = dc.models.layers.DAGGather.from_config(config)

    assert layer_copied.n_graph_feat == layer.n_graph_feat
    assert layer_copied.n_outputs == layer.n_outputs
    assert layer_copied.max_atoms == layer.max_atoms
    assert layer_copied.layer_sizes == layer.layer_sizes
    assert layer_copied.init == layer.init
    assert layer_copied.activation == layer.activation
    assert layer_copied.dropout == layer.dropout


@pytest.mark.tensorflow
def test_message_passing():
    T = 20
    message_fn = 'enn'
    update_fn = 'gru'
    n_hidden = 100
    layer = dc.models.layers.MessagePassing(T, message_fn, update_fn, n_hidden)
    config = layer.get_config()
    layer_copied = dc.models.layers.MessagePassing.from_config(config)

    assert layer_copied.T == layer.T
    assert layer_copied.message_fn == layer.message_fn
    assert layer_copied.update_fn == layer.update_fn
    assert layer_copied.n_hidden == layer.n_hidden


@pytest.mark.tensorflow
def test_edge_network():
    n_pair_features = 8
    n_hidden = 100
    init = 'glorot_uniform'
    layer = dc.models.layers.EdgeNetwork(n_pair_features, n_hidden, init)
    config = layer.get_config()
    layer_copied = dc.models.layers.EdgeNetwork.from_config(config)

    assert layer_copied.n_pair_features == layer.n_pair_features
    assert layer_copied.n_hidden == layer.n_hidden
    assert layer_copied.init == layer.init


@pytest.mark.tensorflow
def test_gru():
    n_hidden = 100
    init = 'glorot_uniform'
    layer = dc.models.layers.GatedRecurrentUnit(n_hidden, init)
    config = layer.get_config()
    layer_copied = dc.models.layers.GatedRecurrentUnit.from_config(config)

    assert layer_copied.n_hidden == layer.n_hidden
    assert layer_copied.init == layer.init


@pytest.mark.tensorflow
def test_set_gather():
    M = 10
    batch_size = 16
    n_hidden = 100
    init = 'orthogonal'

    layer = dc.models.layers.SetGather(M, batch_size, n_hidden, init)
    config = layer.get_config()
    layer_copied = dc.models.layers.SetGather.from_config(config)

    assert layer_copied.M == layer.M
    assert layer_copied.batch_size == layer.batch_size
    assert layer_copied.n_hidden == layer.n_hidden
    assert layer_copied.init == layer.init
