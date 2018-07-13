"""A Model whose structure is defined by an ontology."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import layers
from deepchem.metrics import to_one_hot
import tensorflow as tf


class OntologyNode(object):

  def __init__(self, id, n_outputs, feature_ids=[], children=[]):
    self.id = id
    self.n_outputs = n_outputs
    self.feature_ids = feature_ids
    self.children = children


class OntologyModel(TensorGraph):
  """Implements ontology based models.

  The model is based on Ma et al., "Using deep learning to model the hierarchical
  structure and function of a cell" (https://doi.org/10.1038/nmeth.4627).  The
  model structure is defined by an ontology: a set of features grouped into
  categories, which in turn are arranged hierarchically to form a directed
  acyclic graph.  An example is the Gene Ontology (GO) classifications which
  groups genes into a set of hierarchical categories based on their biological
  role.  Using a known ontology to define the model structure has two benefits.
  First, incorporating prior knowledge can sometimes lead to much more accurate
  predictions for a fixed model size.  Second, it makes the model's results
  much easier to interpret.
  """

  def __init__(self,
               n_tasks,
               feature_ids,
               root_node,
               mode="regression",
               n_classes=2,
               intermediate_loss_weight=0.3,
               weight_decay_penalty=0.0,
               **kwargs):
    super(OntologyModel, self).__init__(**kwargs)
    self.n_tasks = n_tasks
    self.feature_ids = feature_ids
    self.mode = mode
    self.n_classes = n_classes
    self._feature_index = dict((f, i) for i, f in enumerate(feature_ids))
    self._features = layers.Transpose(
        (1, 0), in_layers=layers.Feature(shape=(None, len(feature_ids))))
    self.output_for_node = {}
    self.prediction_for_node = {}
    if mode not in ('regression', 'classification'):
      raise ValueError('Mode must be "regression" or "classification"')

    # Construct layers for all nodes.

    logits_for_node = {}
    self._build_layers(root_node)
    for id in self.output_for_node:
      if mode == 'regression':
        prediction = layers.Dense(
            in_layers=self.output_for_node[id], out_channels=n_tasks)
      else:
        logits = layers.Reshape(
            shape=(-1, n_tasks, n_classes),
            in_layers=layers.Dense(
                in_layers=self.output_for_node[id],
                out_channels=n_tasks * n_classes))
        prediction = layers.SoftMax(logits)
        logits_for_node[id] = logits
      self.prediction_for_node[id] = prediction
      self.add_output(self.prediction_for_node[id])
    self.set_default_outputs([self.prediction_for_node[root_node.id]])

    # Create the loss function.

    losses = []
    loss_weights = []
    weights = layers.Weights(shape=(None, n_tasks))
    if mode == 'regression':
      labels = layers.Label(shape=(None, n_tasks))
      for id in self.prediction_for_node:
        losses.append(
            layers.ReduceSum(
                layers.L2Loss([labels, self.prediction_for_node[id], weights])))
        loss_weights.append(1.0
                            if id == root_node.id else intermediate_loss_weight)
    else:
      labels = layers.Label(shape=(None, n_tasks, n_classes))
      for id in self.prediction_for_node:
        losses.append(
            layers.WeightedError([
                layers.SoftMaxCrossEntropy([labels, logits_for_node[id]]),
                weights
            ]))
        loss_weights.append(1.0
                            if id == root_node.id else intermediate_loss_weight)
    loss = layers.Add(in_layers=losses, weights=loss_weights)
    if weight_decay_penalty != 0.0:
      loss = layers.WeightDecay(weight_decay_penalty, 'l2', in_layers=loss)
    self.set_loss(loss)

  def _build_layers(self, node):
    inputs = []

    # Create inputs for the features.

    if len(node.feature_ids) > 0:
      indices = []
      for f in node.feature_ids:
        if f in self._feature_index:
          indices.append([self._feature_index[f]])
        else:
          raise ValueError('Unknown feature "%s"' % f)
      inputs.append(
          layers.Transpose(
              (1, 0),
              in_layers=layers.Gather(
                  in_layers=self._features, indices=indices)))

    # Create inputs for the children.

    if len(node.children) > 0:
      for child in node.children:
        if child.id not in self.output_for_node:
          self._build_layers(child)
        inputs.append(self.output_for_node[child.id])

    # Concatenate all inputs together.

    if len(inputs) == 0:
      raise ValueError('OntologyNode must have at least one child or feature')
    if len(inputs) == 1:
      inputs = inputs[0]
    else:
      inputs = layers.Concat(inputs)

    # Create the output.

    dense = layers.Dense(
        node.n_outputs, in_layers=inputs, activation_fn=tf.tanh)
    output = layers.BatchNorm(dense)
    self.output_for_node[node.id] = output

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = dict()
        if y_b is not None and not predict:
          if self.mode == 'regression':
            feed_dict[self.labels[0]] = y_b
          else:
            feed_dict[self.labels[0]] = to_one_hot(y_b.flatten(),
                                                   self.n_classes).reshape(
                                                       -1, self.n_tasks,
                                                       self.n_classes)
        if X_b is not None:
          feed_dict[self.features[0]] = X_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        yield feed_dict

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    tensors = {}
    for layer, column in zip(self.features, feature_columns):
      tensors[layer] = tf.feature_column.input_layer(features, [column])
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
          features, [weight_column])
    if labels is not None:
      if self.mode == 'regression':
        tensors[self.labels[0]] = tf.cast(labels, self.labels[0].dtype)
      else:
        tensors[self.labels[0]] = tf.one_hot(
            tf.cast(labels, tf.int32), self.n_classes)
    return tensors
