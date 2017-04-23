import numpy as np
import tensorflow as tf
import six

from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.utils.evaluate import GeneratorEvaluator
from deepchem.models.tensorgraph.layers import Input, BatchNorm, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, Weights, Feature
from deepchem.models.tensorgraph.graph_layers import WeaveLayer, WeaveGather, \
    Combine_AP, Separate_AP, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.trans import undo_transforms


class WeaveTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               n_atom_feat=75,
               n_pair_feat=14,
               n_hidden=50,
               n_graph_feat=128,
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

    """
    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.n_graph_feat = n_graph_feat
    super(WeaveTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """Building graph structures:
    Features => WeaveLayer => WeaveLayer => Dense => WeaveGather => Classification or Regression
    """
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.pair_features = Feature(shape=(None, self.n_pair_feat))
    combined = Combine_AP(in_layers=[self.atom_features, self.pair_features])
    self.pair_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)
    weave_layer1 = WeaveLayer(
        n_atom_input_feat=self.n_atom_feat,
        n_pair_input_feat=self.n_pair_feat,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        in_layers=[combined, self.pair_split, self.atom_to_pair])
    weave_layer2 = WeaveLayer(
        n_atom_input_feat=self.n_hidden,
        n_pair_input_feat=self.n_hidden,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        update_pair=False,
        in_layers=[weave_layer1, self.pair_split, self.atom_to_pair])
    separated = Separate_AP(in_layers=[weave_layer2])
    dense1 = Dense(
        out_channels=self.n_graph_feat,
        activation_fn=tf.nn.relu,
        in_layers=[separated])
    batch_norm1 = BatchNormLayer(in_layers=[dense1])
    weave_gather = WeaveGather(
        self.batch_size,
        n_input=self.n_graph_feat,
        guassian_expand=True,
        in_layers=[batch_norm1, self.atom_split])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      if self.mode == "classification":
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[weave_gather])
        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.labels_fd.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == "regression":
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[weave_gather])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.labels_fd.append(label)
        cost = L2LossLayer(in_layers=[label, regression])
        costs.append(cost)

    all_cost = Concat(in_layers=costs)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    """ TensorGraph style implementation
    similar to deepchem.models.tf_new_models.graph_topology.AlternateWeaveTopology.batch_to_feed_dict
    """
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            if self.mode == "classification":
              feed_dict[label] = to_one_hot(y_b[:, index])
            if self.mode == "regression":
              feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None and not predict:
          feed_dict[self.weights] = w_b

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
                  np.array([C1.flatten() + start, C0.flatten() + start])))
          # number of pairs for each atom
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms

          # atom features
          atom_feat.append(mol.get_atom_features())
          # pair features
          pair_feat.append(
              np.reshape(mol.get_pair_features(), (n_atoms * n_atoms,
                                                   self.n_pair_feat)))

        feed_dict[self.atom_features] = np.concatenate(atom_feat, axis=0)
        feed_dict[self.pair_features] = np.concatenate(pair_feat, axis=0)
        feed_dict[self.pair_split] = np.array(pair_split)
        feed_dict[self.atom_split] = np.array(atom_split)
        feed_dict[self.atom_to_pair] = np.concatenate(atom_to_pair, axis=0)
        yield feed_dict

  def predict(self, dataset, transformers=[], batch_size=None):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers)

  def predict_proba(self, dataset, transformers=[], batch_size=None):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_proba_on_generator(generator, transformers)

  def predict_on_generator(self, generator, transformers=[]):
    retval = self.predict_proba_on_generator(generator, transformers)
    if self.mode == 'classification':
      retval = np.expand_dims(from_one_hot(retval, axis=2), axis=1)
    return retval

  def predict_proba_on_generator(self, generator, transformers=[]):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, self.last_checkpoint)
        out_tensors = [x.out_tensor for x in self.outputs]
        results = []
        for feed_dict in generator:
          feed_dict = {
              self.layers[k.name].out_tensor: v
              for k, v in six.iteritems(feed_dict)
          }
          result = np.array(sess.run(out_tensors, feed_dict=feed_dict))
          if len(result.shape) == 3:
            result = np.transpose(result, axes=[1, 0, 2])
          if len(transformers) > 0:
            result = undo_transforms(result, transformers)
          results.append(result)
        return np.concatenate(results, axis=0)


class DTNNTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               n_embedding=30,
               n_hidden=100,
               n_distance=100,
               distance_min=-1,
               distance_max=18,
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

    """
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
    super(DTNNTensorGraph, self).__init__(**kwargs)
    assert self.mode == "regression"
    self.build_graph()

  def build_graph(self):
    """Building graph structures:
    Features => DTNNEmbedding => DTNNStep => DTNNStep => DTNNGather => Regression
    """
    self.atom_number = Feature(shape=(None,), dtype=tf.int32)
    self.distance = Feature(shape=(None, self.n_distance))
    self.atom_membership = Feature(shape=(None,), dtype=tf.int32)
    self.distance_membership_i = Feature(shape=(None,), dtype=tf.int32)
    self.distance_membership_j = Feature(shape=(None,), dtype=tf.int32)

    dtnn_embedding = DTNNEmbedding(
        n_embedding=self.n_embedding, in_layers=[self.atom_number])
    dtnn_layer1 = DTNNStep(
        n_embedding=self.n_embedding,
        n_distance=self.n_distance,
        in_layers=[
            dtnn_embedding, self.distance, self.distance_membership_i,
            self.distance_membership_j
        ])
    dtnn_layer2 = DTNNStep(
        n_embedding=self.n_embedding,
        n_distance=self.n_distance,
        in_layers=[
            dtnn_layer1, self.distance, self.distance_membership_i,
            self.distance_membership_j
        ])
    dtnn_gather = DTNNGather(
        n_embedding=self.n_embedding,
        n_outputs=self.n_hidden,
        in_layers=[dtnn_layer2, self.atom_membership])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      regression = Dense(
          out_channels=1, activation_fn=None, in_layers=[dtnn_gather])
      self.add_output(regression)

      label = Label(shape=(None, 1))
      self.labels_fd.append(label)
      cost = L2LossLayer(in_layers=[label, regression])
      costs.append(cost)

    all_cost = Concat(in_layers=costs)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    """ TensorGraph style implementation
    similar to deepchem.models.tf_new_models.graph_topology.DTNNGraphTopology.batch_to_feed_dict
    """
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None and not predict:
          feed_dict[self.weights] = w_b

        distance = []
        atom_membership = []
        distance_membership_i = []
        distance_membership_j = []
        num_atoms = list(map(sum, X_b.astype(bool)[:, :, 0]))
        atom_number = [
            np.round(
                np.power(2 * np.diag(X_b[i, :num_atoms[i], :num_atoms[i]]), 1 /
                         2.4)).astype(int) for i in range(len(num_atoms))
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
        feed_dict[self.atom_number] = np.concatenate(atom_number)
        distance = np.concatenate(distance, 0)
        feed_dict[self.distance] = np.exp(-np.square(distance - self.steps) /
                                          (2 * self.step_size**2))
        feed_dict[self.distance_membership_i] = np.concatenate(
            distance_membership_i)
        feed_dict[self.distance_membership_j] = np.concatenate(
            distance_membership_j)
        feed_dict[self.atom_membership] = np.concatenate(atom_membership)

        yield feed_dict

  def predict(self, dataset, transformers=[], batch_size=None):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers)

  def predict_proba(self, dataset, transformers=[], batch_size=None):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_proba_on_generator(generator, transformers)

  def predict_on_generator(self, generator, transformers=[]):
    retval = self.predict_proba_on_generator(generator, transformers)
    if self.mode == 'classification':
      retval = np.expand_dims(from_one_hot(retval, axis=2), axis=1)
    return retval

  def predict_proba_on_generator(self, generator, transformers=[]):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, self.last_checkpoint)
        out_tensors = [x.out_tensor for x in self.outputs]
        results = []
        for feed_dict in generator:
          feed_dict = {
              self.layers[k.name].out_tensor: v
              for k, v in six.iteritems(feed_dict)
          }
          result = np.array(sess.run(out_tensors, feed_dict=feed_dict))
          if len(result.shape) == 3:
            result = np.transpose(result, axes=[1, 0, 2])
          if len(transformers) > 0:
            result = undo_transforms(result, transformers)
          results.append(result)
        return np.concatenate(results, axis=0)


class DAGTensorGraph(TensorGraph):

  def __init__(self,
               n_tasks,
               max_atoms=50,
               n_atom_feat=75,
               n_graph_feat=30,
               n_outputs=30,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset
    n_atom_feat: int, optional
      Number of features per atom.
    n_graph_feat: int, optional
      Number of features for atom in the graph
    n_outputs: int, optional
      Number of features for each molecule

    """
    self.n_tasks = n_tasks
    self.max_atoms = max_atoms
    self.n_atom_feat = n_atom_feat
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs
    super(DAGTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """Building graph structures:
    Features => DAGLayer => DAGGather => Classification or Regression
    """
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.parents = Feature(
        shape=(None, self.max_atoms, self.max_atoms), dtype=tf.int32)
    self.calculation_orders = Feature(
        shape=(None, self.max_atoms), dtype=tf.int32)
    self.calculation_masks = Feature(
        shape=(None, self.max_atoms), dtype=tf.bool)
    self.membership = Feature(shape=(None,), dtype=tf.int32)
    self.n_atoms = Feature(shape=(), dtype=tf.int32)
    dag_layer1 = DAGLayer(
        n_graph_feat=self.n_graph_feat,
        n_atom_feat=self.n_atom_feat,
        max_atoms=self.max_atoms,
        batch_size=self.batch_size,
        in_layers=[
            self.atom_features, self.parents, self.calculation_orders,
            self.calculation_masks, self.n_atoms
        ])
    dag_gather = DAGGather(
        n_graph_feat=self.n_graph_feat,
        n_outputs=self.n_outputs,
        max_atoms=self.max_atoms,
        in_layers=[dag_layer1, self.membership])

    costs = []
    self.labels_fd = []
    for task in range(self.n_tasks):
      if self.mode == "classification":
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[dag_gather])
        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.labels_fd.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == "regression":
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[dag_gather])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.labels_fd.append(label)
        cost = L2LossLayer(in_layers=[label, regression])
        costs.append(cost)

    all_cost = Concat(in_layers=costs)
    self.weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[all_cost, self.weights])
    self.set_loss(loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):
    """ TensorGraph style implementation
    similar to deepchem.models.tf_new_models.graph_topology.DAGGraphTopology.batch_to_feed_dict
    """
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):

        feed_dict = dict()
        if y_b is not None and not predict:
          for index, label in enumerate(self.labels_fd):
            if self.mode == "classification":
              feed_dict[label] = to_one_hot(y_b[:, index])
            if self.mode == "regression":
              feed_dict[label] = y_b[:, index:index + 1]
        if w_b is not None and not predict:
          feed_dict[self.weights] = w_b

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

        feed_dict[self.atom_features] = np.concatenate(atoms_all, axis=0)
        feed_dict[self.parents] = np.stack(parents_all, axis=0)
        feed_dict[self.calculation_orders] = np.concatenate(
            calculation_orders, axis=0)
        feed_dict[self.calculation_masks] = np.concatenate(
            calculation_masks, axis=0)
        feed_dict[self.membership] = np.array(membership)
        feed_dict[self.n_atoms] = n_atoms
        yield feed_dict

  def predict(self, dataset, transformers=[], batch_size=None):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers)

  def predict_proba(self, dataset, transformers=[], batch_size=None):
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_proba_on_generator(generator, transformers)

  def predict_on_generator(self, generator, transformers=[]):
    retval = self.predict_proba_on_generator(generator, transformers)
    if self.mode == 'classification':
      retval = np.expand_dims(from_one_hot(retval, axis=2), axis=1)
    return retval

  def predict_proba_on_generator(self, generator, transformers=[]):
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, self.last_checkpoint)
        out_tensors = [x.out_tensor for x in self.outputs]
        results = []
        for feed_dict in generator:
          feed_dict = {
              self.layers[k.name].out_tensor: v
              for k, v in six.iteritems(feed_dict)
          }
          result = np.array(sess.run(out_tensors, feed_dict=feed_dict))
          if len(result.shape) == 3:
            result = np.transpose(result, axes=[1, 0, 2])
          if len(transformers) > 0:
            result = undo_transforms(result, transformers)
          results.append(result)
        return np.concatenate(results, axis=0)
