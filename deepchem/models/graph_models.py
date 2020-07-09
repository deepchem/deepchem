import collections

import deepchem as dc
import numpy as np
import tensorflow as tf

from deepchem.data import NumpyDataset, pad_features
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.trans import undo_transforms
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization


class TrimGraphOutput(tf.keras.layers.Layer):
  """Trim the output to the correct number of samples.

  GraphGather always outputs fixed size batches.  This layer trims the output
  to the number of samples that were in the actual input tensors.
  """

  def __init__(self, **kwargs):
    super(TrimGraphOutput, self).__init__(**kwargs)

  def call(self, inputs):
    n_samples = tf.squeeze(inputs[1])
    return inputs[0][0:n_samples]


class WeaveModel(KerasModel):
  """Implements Google-style Weave Graph Convolutions

  This model implements the Weave style graph convolutions
  from the following paper.

  Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints." Journal of computer-aided molecular design 30.8 (2016): 595-608.

  The biggest difference between WeaveModel style convolutions
  and GraphConvModel style convolutions is that Weave
  convolutions model bond features explicitly. This has the
  side effect that it needs to construct a NxN matrix
  explicitly to model bond interactions. This may cause
  scaling issues, but may possibly allow for better modeling
  of subtle bond effects.
  """

  def __init__(self,
               n_tasks,
               n_atom_feat=75,
               n_pair_feat=14,
               n_hidden=50,
               n_graph_feat=128,
               mode="classification",
               n_classes=2,
               batch_size=100,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    n_hidden: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    n_graph_feat: int, optional
      Number of output features for each molecule(graph)
    mode: str
      Either "classification" or "regression" for type of model.
    n_classes: int
      Number of classes to predict (only used in classification mode)
    """
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")
    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.n_graph_feat = n_graph_feat
    self.mode = mode
    self.n_classes = n_classes

    # Build the model.

    atom_features = Input(shape=(self.n_atom_feat,))
    pair_features = Input(shape=(self.n_pair_feat,))
    pair_split = Input(shape=tuple(), dtype=tf.int32)
    atom_split = Input(shape=tuple(), dtype=tf.int32)
    atom_to_pair = Input(shape=(2,), dtype=tf.int32)
    weave_layer1A, weave_layer1P = layers.WeaveLayer(
        n_atom_input_feat=self.n_atom_feat,
        n_pair_input_feat=self.n_pair_feat,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden)(
            [atom_features, pair_features, pair_split, atom_to_pair])
    weave_layer2A, weave_layer2P = layers.WeaveLayer(
        n_atom_input_feat=self.n_hidden,
        n_pair_input_feat=self.n_hidden,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        update_pair=False)(
            [weave_layer1A, weave_layer1P, pair_split, atom_to_pair])
    dense1 = Dense(self.n_graph_feat, activation=tf.nn.tanh)(weave_layer2A)
    # Batch normalization causes issues, spitting out NaNs if
    # allowed to train
    batch_norm1 = BatchNormalization(epsilon=1e-5, trainable=False)(dense1)
    weave_gather = layers.WeaveGather(
        batch_size, n_input=self.n_graph_feat,
        gaussian_expand=True)([batch_norm1, atom_split])

    n_tasks = self.n_tasks
    if self.mode == 'classification':
      n_classes = self.n_classes
      logits = Reshape((n_tasks,
                        n_classes))(Dense(n_tasks * n_classes)(weave_gather))
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = SoftmaxCrossEntropy()
    else:
      output = Dense(n_tasks)(weave_gather)
      outputs = [output]
      output_types = ['prediction']
      loss = L2Loss()
    model = tf.keras.Model(
        inputs=[
            atom_features, pair_features, pair_split, atom_split, atom_to_pair
        ],
        outputs=outputs)
    super(WeaveModel, self).__init__(
        model, loss, output_types=output_types, batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if y_b is not None:
          if self.mode == 'classification':
            y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                -1, self.n_tasks, self.n_classes)
        atom_feat = []
        pair_feat = []
        atom_split = []
        atom_to_pair = []
        pair_split = []
        start = 0
        for im, mol in enumerate(X_b):
          n_atoms = mol.get_num_atoms()
          # number of atoms in each molecule
          atom_split.extend([im] * n_atoms)
          # index of pair features
          C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
          atom_to_pair.append(
              np.transpose(
                  np.array([C1.flatten() + start,
                            C0.flatten() + start])))
          # number of pairs for each atom
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms

          # atom features
          atom_feat.append(mol.get_atom_features())
          # pair features
          pair_feat.append(
              np.reshape(mol.get_pair_features(),
                         (n_atoms * n_atoms, self.n_pair_feat)))

        inputs = [
            np.concatenate(atom_feat, axis=0),
            np.concatenate(pair_feat, axis=0),
            np.array(pair_split),
            np.array(atom_split),
            np.concatenate(atom_to_pair, axis=0)
        ]
        yield (inputs, [y_b], [w_b])


class DTNNModel(KerasModel):
  """Deep Tensor Neural Networks

  This class implements deep tensor neural networks as first defined in

  Schütt, Kristof T., et al. "Quantum-chemical insights from deep tensor neural networks." Nature communications 8.1 (2017): 1-8.
  """

  def __init__(self,
               n_tasks,
               n_embedding=30,
               n_hidden=100,
               n_distance=100,
               distance_min=-1,
               distance_max=18,
               output_activation=True,
               mode="regression",
               dropout=0.0,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_embedding: int, optional
      Number of features per atom.
    n_hidden: int, optional
      Number of features for each molecule after DTNNStep
    n_distance: int, optional
      granularity of distance matrix
      step size will be (distance_max-distance_min)/n_distance
    distance_min: float, optional
      minimum distance of atom pairs, default = -1 Angstorm
    distance_max: float, optional
      maximum distance of atom pairs, default = 18 Angstorm
    mode: str
      Only "regression" is currently supported.
    dropout: float
      the dropout probablity to use.
    """
    if mode not in ['regression']:
      raise ValueError("Only 'regression' mode is currently supported")
    self.n_tasks = n_tasks
    self.n_embedding = n_embedding
    self.n_hidden = n_hidden
    self.n_distance = n_distance
    self.distance_min = distance_min
    self.distance_max = distance_max
    self.step_size = (distance_max - distance_min) / n_distance
    self.steps = np.array(
        [distance_min + i * self.step_size for i in range(n_distance)])
    self.steps = np.expand_dims(self.steps, 0)
    self.output_activation = output_activation
    self.mode = mode
    self.dropout = dropout

    # Build the model.

    atom_number = Input(shape=tuple(), dtype=tf.int32)
    distance = Input(shape=(self.n_distance,))
    atom_membership = Input(shape=tuple(), dtype=tf.int32)
    distance_membership_i = Input(shape=tuple(), dtype=tf.int32)
    distance_membership_j = Input(shape=tuple(), dtype=tf.int32)

    dtnn_embedding = layers.DTNNEmbedding(
        n_embedding=self.n_embedding)(atom_number)
    if self.dropout > 0.0:
      dtnn_embedding = Dropout(rate=self.dropout)(dtnn_embedding)
    dtnn_layer1 = layers.DTNNStep(
        n_embedding=self.n_embedding, n_distance=self.n_distance)([
            dtnn_embedding, distance, distance_membership_i,
            distance_membership_j
        ])
    if self.dropout > 0.0:
      dtnn_layer1 = Dropout(rate=self.dropout)(dtnn_layer1)
    dtnn_layer2 = layers.DTNNStep(
        n_embedding=self.n_embedding, n_distance=self.n_distance)([
            dtnn_layer1, distance, distance_membership_i, distance_membership_j
        ])
    if self.dropout > 0.0:
      dtnn_layer2 = Dropout(rate=self.dropout)(dtnn_layer2)
    dtnn_gather = layers.DTNNGather(
        n_embedding=self.n_embedding,
        layer_sizes=[self.n_hidden],
        n_outputs=self.n_tasks,
        output_activation=self.output_activation)(
            [dtnn_layer2, atom_membership])
    if self.dropout > 0.0:
      dtnn_gather = Dropout(rate=self.dropout)(dtnn_gather)

    n_tasks = self.n_tasks
    output = Dense(n_tasks)(dtnn_gather)
    model = tf.keras.Model(
        inputs=[
            atom_number, distance, atom_membership, distance_membership_i,
            distance_membership_j
        ],
        outputs=[output])
    super(DTNNModel, self).__init__(model, L2Loss(), **kwargs)

  def compute_features_on_batch(self, X_b):
    """Computes the values for different Feature Layers on given batch

    A tf.py_func wrapper is written around this when creating the
    input_fn for tf.Estimator

    """
    distance = []
    atom_membership = []
    distance_membership_i = []
    distance_membership_j = []
    num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))
    atom_number = [
        np.round(
            np.power(2 * np.diag(X_b[i, :num_atoms[i], :num_atoms[i]]),
                     1 / 2.4)).astype(int) for i in range(len(num_atoms))
    ]
    start = 0
    for im, molecule in enumerate(atom_number):
      distance_matrix = np.outer(
          molecule, molecule) / X_b[im, :num_atoms[im], :num_atoms[im]]
      np.fill_diagonal(distance_matrix, -100)
      distance.append(np.expand_dims(distance_matrix.flatten(), 1))
      atom_membership.append([im] * num_atoms[im])
      membership = np.array([np.arange(num_atoms[im])] * num_atoms[im])
      membership_i = membership.flatten(order='F')
      membership_j = membership.flatten()
      distance_membership_i.append(membership_i + start)
      distance_membership_j.append(membership_j + start)
      start = start + num_atoms[im]

    atom_number = np.concatenate(atom_number).astype(np.int32)
    distance = np.concatenate(distance, axis=0)
    gaussian_dist = np.exp(
        -np.square(distance - self.steps) / (2 * self.step_size**2))
    gaussian_dist = gaussian_dist.astype(np.float32)
    atom_mem = np.concatenate(atom_membership).astype(np.int32)
    dist_mem_i = np.concatenate(distance_membership_i).astype(np.int32)
    dist_mem_j = np.concatenate(distance_membership_j).astype(np.int32)

    features = [atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j]

    return features

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        yield (self.compute_features_on_batch(X_b), [y_b], [w_b])


class DAGModel(KerasModel):
  """Directed Acyclic Graph models for molecular property prediction.

    This model is based on the following paper: 

    Lusci, Alessandro, Gianluca Pollastri, and Pierre Baldi. "Deep architectures and deep learning in chemoinformatics: the prediction of aqueous solubility for drug-like molecules." Journal of chemical information and modeling 53.7 (2013): 1563-1575.

   The basic idea for this paper is that a molecule is usually
   viewed as an undirected graph. However, you can convert it to
   a series of directed graphs. The idea is that for each atom,
   you make a DAG using that atom as the vertex of the DAG and
   edges pointing "inwards" to it. This transformation is
   implemented in
   `dc.trans.transformers.DAGTransformer.UG_to_DAG`.

   This model accepts ConvMols as input, just as GraphConvModel
   does, but these ConvMol objects must be transformed by
   dc.trans.DAGTransformer. 

   As a note, performance of this model can be a little
   sensitive to initialization. It might be worth training a few
   different instantiations to get a stable set of parameters.
   """

  def __init__(self,
               n_tasks,
               max_atoms=50,
               n_atom_feat=75,
               n_graph_feat=30,
               n_outputs=30,
               layer_sizes=[100],
               layer_sizes_gather=[100],
               dropout=None,
               mode="classification",
               n_classes=2,
               uncertainty=False,
               batch_size=100,
               **kwargs):
    """   
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset.
    n_atom_feat: int, optional
      Number of features per atom.
    n_graph_feat: int, optional
      Number of features for atom in the graph.
    n_outputs: int, optional
      Number of features for each molecule.
    layer_sizes: list of int, optional
      List of hidden layer size(s) in the propagation step:
      length of this list represents the number of hidden layers,
      and each element is the width of corresponding hidden layer.
    layer_sizes_gather: list of int, optional
      List of hidden layer size(s) in the gather step.
    dropout: None or float, optional
      Dropout probability, applied after each propagation step and gather step.
    mode: str, optional
      Either "classification" or "regression" for type of model.
    n_classes: int
      the number of classes to predict (only used in classification mode)
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    """
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")
    self.n_tasks = n_tasks
    self.max_atoms = max_atoms
    self.n_atom_feat = n_atom_feat
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs
    self.layer_sizes = layer_sizes
    self.layer_sizes_gather = layer_sizes_gather
    self.dropout = dropout
    self.mode = mode
    self.n_classes = n_classes
    self.uncertainty = uncertainty
    if uncertainty:
      if mode != "regression":
        raise ValueError("Uncertainty is only supported in regression mode")
      if dropout is None or dropout == 0.0:
        raise ValueError('Dropout must be included to predict uncertainty')

    ############################################
    print("self.dropout")
    print(self.dropout)
    ############################################
    # Build the model.

    atom_features = Input(shape=(self.n_atom_feat,))
    parents = Input(shape=(self.max_atoms, self.max_atoms), dtype=tf.int32)
    calculation_orders = Input(shape=(self.max_atoms,), dtype=tf.int32)
    calculation_masks = Input(shape=(self.max_atoms,), dtype=tf.bool)
    membership = Input(shape=tuple(), dtype=tf.int32)
    n_atoms = Input(shape=tuple(), dtype=tf.int32)
    dag_layer1 = layers.DAGLayer(
        n_graph_feat=self.n_graph_feat,
        n_atom_feat=self.n_atom_feat,
        max_atoms=self.max_atoms,
        layer_sizes=self.layer_sizes,
        dropout=self.dropout,
        batch_size=batch_size)([
            atom_features, parents, calculation_orders, calculation_masks,
            n_atoms
        ])
    dag_gather = layers.DAGGather(
        n_graph_feat=self.n_graph_feat,
        n_outputs=self.n_outputs,
        max_atoms=self.max_atoms,
        layer_sizes=self.layer_sizes_gather,
        dropout=self.dropout)([dag_layer1, membership])
    n_tasks = self.n_tasks
    if self.mode == 'classification':
      n_classes = self.n_classes
      logits = Reshape((n_tasks,
                        n_classes))(Dense(n_tasks * n_classes)(dag_gather))
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = SoftmaxCrossEntropy()
    else:
      output = Dense(n_tasks)(dag_gather)
      if self.uncertainty:
        log_var = Dense(n_tasks)(dag_gather)
        var = Activation(tf.exp)(log_var)
        outputs = [output, var, output, log_var]
        output_types = ['prediction', 'variance', 'loss', 'loss']

        def loss(outputs, labels, weights):
          diff = labels[0] - outputs[0]
          return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
      else:
        outputs = [output]
        output_types = ['prediction']
        loss = L2Loss()
    model = tf.keras.Model(
        inputs=[
            atom_features,
            parents,
            calculation_orders,
            calculation_masks,
            membership,
            n_atoms  #, dropout_switch
        ],
        outputs=outputs)
    super(DAGModel, self).__init__(
        model, loss, output_types=output_types, batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    """TensorGraph style implementation"""
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        if y_b is not None and self.mode == 'classification':
          y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
              -1, self.n_tasks, self.n_classes)

        atoms_per_mol = [mol.get_num_atoms() for mol in X_b]
        n_atoms = sum(atoms_per_mol)
        start_index = [0] + list(np.cumsum(atoms_per_mol)[:-1])

        atoms_all = []
        # calculation orders for a batch of molecules
        parents_all = []
        calculation_orders = []
        calculation_masks = []
        membership = []
        for idm, mol in enumerate(X_b):
          # padding atom features vector of each molecule with 0
          atoms_all.append(mol.get_atom_features())
          parents = mol.parents
          parents_all.extend(parents)
          calculation_index = np.array(parents)[:, :, 0]
          mask = np.array(calculation_index - self.max_atoms, dtype=bool)
          calculation_orders.append(calculation_index + start_index[idm])
          calculation_masks.append(mask)
          membership.extend([idm] * atoms_per_mol[idm])
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)

        yield ([
            np.concatenate(atoms_all, axis=0),
            np.stack(parents_all, axis=0),
            np.concatenate(calculation_orders, axis=0),
            np.concatenate(calculation_masks, axis=0),
            np.array(membership),
            np.array(n_atoms), dropout
        ], [y_b], [w_b])


class _GraphConvKerasModel(tf.keras.Model):

  def __init__(self,
               n_tasks,
               graph_conv_layers,
               dense_layer_size=128,
               dropout=0.0,
               mode="classification",
               number_atom_features=75,
               n_classes=2,
               batch_normalize=True,
               uncertainty=False,
               batch_size=100):
    """An internal keras model class.

    The graph convolutions use a nonstandard control flow so the
    standard Keras functional API can't support them. We instead
    use the imperative "subclassing" API to implement the graph
    convolutions.

    All arguments have the same meaning as in GraphConvModel.
    """
    super(_GraphConvKerasModel, self).__init__()
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    self.mode = mode
    self.uncertainty = uncertainty

    if not isinstance(dropout, collections.Sequence):
      dropout = [dropout] * (len(graph_conv_layers) + 1)
    if len(dropout) != len(graph_conv_layers) + 1:
      raise ValueError('Wrong number of dropout probabilities provided')
    if uncertainty:
      if mode != "regression":
        raise ValueError("Uncertainty is only supported in regression mode")
      if any(d == 0.0 for d in dropout):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

    self.graph_convs = [
        layers.GraphConv(layer_size, activation_fn=tf.nn.relu)
        for layer_size in graph_conv_layers
    ]
    self.batch_norms = [
        BatchNormalization(fused=False) if batch_normalize else None
        for _ in range(len(graph_conv_layers) + 1)
    ]
    self.dropouts = [
        Dropout(rate=rate) if rate > 0.0 else None for rate in dropout
    ]
    self.graph_pools = [layers.GraphPool() for _ in graph_conv_layers]
    self.dense = Dense(dense_layer_size, activation=tf.nn.relu)
    self.graph_gather = layers.GraphGather(
        batch_size=batch_size, activation_fn=tf.nn.tanh)
    self.trim = TrimGraphOutput()
    if self.mode == 'classification':
      self.reshape_dense = Dense(n_tasks * n_classes)
      self.reshape = Reshape((n_tasks, n_classes))
      self.softmax = Softmax()
    else:
      self.regression_dense = Dense(n_tasks)
      if self.uncertainty:
        self.uncertainty_dense = Dense(n_tasks)
        self.uncertainty_trim = TrimGraphOutput()
        self.uncertainty_activation = Activation(tf.exp)

  def call(self, inputs, training=False):
    atom_features = inputs[0]
    degree_slice = tf.cast(inputs[1], dtype=tf.int32)
    membership = tf.cast(inputs[2], dtype=tf.int32)
    n_samples = tf.cast(inputs[3], dtype=tf.int32)
    deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]

    in_layer = atom_features
    for i in range(len(self.graph_convs)):
      gc_in = [in_layer, degree_slice, membership] + deg_adjs
      gc1 = self.graph_convs[i](gc_in)
      if self.batch_norms[i] is not None:
        gc1 = self.batch_norms[i](gc1, training=training)
      if training and self.dropouts[i] is not None:
        gc1 = self.dropouts[i](gc1, training=training)
      gp_in = [gc1, degree_slice, membership] + deg_adjs
      in_layer = self.graph_pools[i](gp_in)
    dense = self.dense(in_layer)
    if self.batch_norms[-1] is not None:
      dense = self.batch_norms[-1](dense, training=training)
    if training and self.dropouts[-1] is not None:
      dense = self.dropouts[-1](dense, training=training)
    neural_fingerprint = self.graph_gather([dense, degree_slice, membership] +
                                           deg_adjs)
    if self.mode == 'classification':
      logits = self.reshape(self.reshape_dense(neural_fingerprint))
      logits = self.trim([logits, n_samples])
      output = self.softmax(logits)
      outputs = [output, logits, neural_fingerprint]
    else:
      output = self.regression_dense(neural_fingerprint)
      output = self.trim([output, n_samples])
      if self.uncertainty:
        log_var = self.uncertainty_dense(neural_fingerprint)
        log_var = self.uncertainty_trim([log_var, n_samples])
        var = self.uncertainty_activation(log_var)
        outputs = [output, var, output, log_var, neural_fingerprint]
      else:
        outputs = [output, neural_fingerprint]

    return outputs


class GraphConvModel(KerasModel):
  """Graph Convolutional Models.

  This class implements the graph convolutional model from the
  following paper:


  Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints." Advances in neural information processing systems. 2015.

  """

  def __init__(self,
               n_tasks,
               graph_conv_layers=[64, 64],
               dense_layer_size=128,
               dropout=0.0,
               mode="classification",
               number_atom_features=75,
               n_classes=2,
               batch_size=100,
               batch_normalize=True,
               uncertainty=False,
               **kwargs):
    """The wrapper class for graph convolutions.

    Note that since the underlying _GraphConvKerasModel class is
    specified using imperative subclassing style, this model
    cannout make predictions for arbitrary outputs. 

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    graph_conv_layers: list of int
      Width of channels for the Graph Convolution Layers
    dense_layer_size: int
      Width of channels for Atom Level Dense Layer before GraphPool
    dropout: list or float
      the dropout probablity to use for each layer.  The length of this list should equal
      len(graph_conv_layers)+1 (one value for each convolution layer, and one for the
      dense layer).  Alternatively this may be a single value instead of a list, in which
      case the same value is used for every layer.
    mode: str
      Either "classification" or "regression"
    number_atom_features: int
        75 is the default number of atom features created, but
        this can vary if various options are passed to the
        function atom_features in graph_features
    n_classes: int
      the number of classes to predict (only used in classification mode)
    batch_normalize: True
      if True, apply batch normalization to model
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    """
    self.mode = mode
    self.n_tasks = n_tasks
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.uncertainty = uncertainty
    model = _GraphConvKerasModel(
        n_tasks,
        graph_conv_layers=graph_conv_layers,
        dense_layer_size=dense_layer_size,
        dropout=dropout,
        mode=mode,
        number_atom_features=number_atom_features,
        n_classes=n_classes,
        batch_normalize=batch_normalize,
        uncertainty=uncertainty,
        batch_size=batch_size)
    if mode == "classification":
      output_types = ['prediction', 'loss', 'embedding']
      loss = SoftmaxCrossEntropy()
    else:
      if self.uncertainty:
        output_types = ['prediction', 'variance', 'loss', 'loss', 'embedding']

        def loss(outputs, labels, weights):
          diff = labels[0] - outputs[0]
          return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
      else:
        output_types = ['prediction', 'embedding']
        loss = L2Loss()
    super(GraphConvModel, self).__init__(
        model, loss, output_types=output_types, batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if self.mode == 'classification':
          y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
              -1, self.n_tasks, self.n_classes)
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        n_samples = np.array(X_b.shape[0])
        inputs = [
            multiConvMol.get_atom_features(), multiConvMol.deg_slice,
            np.array(multiConvMol.membership), n_samples
        ]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        yield (inputs, [y_b], [w_b])


class MPNNModel(KerasModel):
  """ Message Passing Neural Network,

  Message Passing Neural Networks treat graph convolutional
  operations as an instantiation of a more general message
  passing schem. Recall that message passing in a graph is when
  nodes in a graph send each other "messages" and update their
  internal state as a consequence of these messages.

  Ordering structures in this model are built according to


Vinyals, Oriol, Samy Bengio, and Manjunath Kudlur. "Order matters: Sequence to sequence for sets." arXiv preprint arXiv:1511.06391 (2015).

  """

  def __init__(self,
               n_tasks,
               n_atom_feat=70,
               n_pair_feat=8,
               n_hidden=100,
               T=5,
               M=10,
               mode="regression",
               dropout=0.0,
               n_classes=2,
               uncertainty=False,
               batch_size=100,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    n_hidden: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    n_graph_feat: int, optional
      Number of output features for each molecule(graph)
    dropout: float
      the dropout probablity to use.
    n_classes: int
      the number of classes to predict (only used in classification mode)
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    """
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")
    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.T = T
    self.M = M
    self.mode = mode
    self.n_classes = n_classes
    self.uncertainty = uncertainty
    if uncertainty:
      if mode != "regression":
        raise ValueError("Uncertainty is only supported in regression mode")
      if dropout == 0.0:
        raise ValueError('Dropout must be included to predict uncertainty')

    # Build the model.

    atom_features = Input(shape=(self.n_atom_feat,))
    pair_features = Input(shape=(self.n_pair_feat,))
    atom_split = Input(shape=tuple(), dtype=tf.int32)
    atom_to_pair = Input(shape=(2,), dtype=tf.int32)
    n_samples = Input(shape=tuple(), dtype=tf.int32)

    message_passing = layers.MessagePassing(
        self.T, message_fn='enn', update_fn='gru',
        n_hidden=self.n_hidden)([atom_features, pair_features, atom_to_pair])

    atom_embeddings = Dense(self.n_hidden)(message_passing)

    mol_embeddings = layers.SetGather(
        self.M, batch_size,
        n_hidden=self.n_hidden)([atom_embeddings, atom_split])

    dense1 = Dense(2 * self.n_hidden, activation=tf.nn.relu)(mol_embeddings)

    n_tasks = self.n_tasks
    if self.mode == 'classification':
      n_classes = self.n_classes
      logits = Reshape((n_tasks, n_classes))(Dense(n_tasks * n_classes)(dense1))
      logits = TrimGraphOutput()([logits, n_samples])
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss = SoftmaxCrossEntropy()
    else:
      output = Dense(n_tasks)(dense1)
      output = TrimGraphOutput()([output, n_samples])
      if self.uncertainty:
        log_var = Dense(n_tasks)(dense1)
        log_var = TrimGraphOutput()([log_var, n_samples])
        var = Activation(tf.exp)(log_var)
        outputs = [output, var, output, log_var]
        output_types = ['prediction', 'variance', 'loss', 'loss']

        def loss(outputs, labels, weights):
          diff = labels[0] - outputs[0]
          return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
      else:
        outputs = [output]
        output_types = ['prediction']
        loss = L2Loss()
    model = tf.keras.Model(
        inputs=[
            atom_features, pair_features, atom_split, atom_to_pair, n_samples
        ],
        outputs=outputs)
    super(MPNNModel, self).__init__(
        model, loss, output_types=output_types, batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        n_samples = np.array(X_b.shape[0])
        X_b = pad_features(self.batch_size, X_b)
        if y_b is not None and self.mode == 'classification':
          y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
              -1, self.n_tasks, self.n_classes)

        atom_feat = []
        pair_feat = []
        atom_split = []
        atom_to_pair = []
        pair_split = []
        start = 0
        for im, mol in enumerate(X_b):
          n_atoms = mol.get_num_atoms()
          # number of atoms in each molecule
          atom_split.extend([im] * n_atoms)
          # index of pair features
          C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
          atom_to_pair.append(
              np.transpose(
                  np.array([C1.flatten() + start,
                            C0.flatten() + start])))
          # number of pairs for each atom
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms

          # atom features
          atom_feat.append(mol.get_atom_features())
          # pair features
          pair_feat.append(
              np.reshape(mol.get_pair_features(),
                         (n_atoms * n_atoms, self.n_pair_feat)))

        inputs = [
            np.concatenate(atom_feat, axis=0),
            np.concatenate(pair_feat, axis=0),
            np.array(atom_split),
            np.concatenate(atom_to_pair, axis=0), n_samples
        ]
        yield (inputs, [y_b], [w_b])


#################### Deprecation warnings for renamed TensorGraph models ####################

import warnings

TENSORGRAPH_DEPRECATION = "{} is deprecated and has been renamed to {} and will be removed in DeepChem 3.0."


class GraphConvTensorGraph(GraphConvModel):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("GraphConvTensorGraph",
                                       "GraphConvModel"), FutureWarning)

    super(GraphConvTensorGraph, self).__init__(*args, **kwargs)


class WeaveTensorGraph(WeaveModel):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("WeaveTensorGraph", "WeaveModel"),
        FutureWarning)

    super(WeaveModel, self).__init__(*args, **kwargs)


class DTNNTensorGraph(DTNNModel):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("DTNNTensorGraph", "DTNNModel"),
        FutureWarning)

    super(DTNNModel, self).__init__(*args, **kwargs)


class DAGTensorGraph(DAGModel):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("DAGTensorGraph", "DAGModel"),
        FutureWarning)

    super(DAGModel, self).__init__(*args, **kwargs)


class MPNNTensorGraph(MPNNModel):

  def __init__(self, *args, **kwargs):

    warnings.warn(
        TENSORGRAPH_DEPRECATION.format("MPNNTensorGraph", "MPNNModel"),
        FutureWarning)

    super(MPNNModel, self).__init__(*args, **kwargs)
