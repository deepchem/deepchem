"""A Model whose structure is defined by an ontology."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import layers
from deepchem.metrics import to_one_hot
from deepchem.utils import get_data_dir, download_url
import tensorflow as tf
import math
import os


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

  To use this model, you must provide an ontology represented as a tree of
  OntologyNode objects.  Each node corresponds to a category in the ontology.
  It defines the list of features (e.g. genes) that correspond to that category,
  as well as its child nodes (subcategories).  In addition, every feature and
  every node has a unique string identifier that can be used to refer to it.

  As an alternative to building the ontology yourself, you can use the
  create_gene_ontology() function to build a representation of the GO hierarchy.
  It downloads a definition of the hierarchy from the GO website, parses it,
  builds OntologyNodes for all the categories, and returns a root node that you
  can pass to the OntologyModel constructor.

  An important feature of this model is that the outputs of its internal layers
  are meaningful.  During training, it tries to make each category independently
  predict the labels.  By default, predict() returns the predictions for the
  root node of the hierarchy.  You can use the prediction_for_node field to get
  the output layer corresponding to a particular category:

  prediction = model.predict(dataset, outputs=model.prediction_for_node[node_id])
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
    """Create an OntologyModel.

    In addition to the following arguments, this class also accepts
    all the keyword arguments from TensorGraph.

    Parameters
    ----------
    n_tasks: int
      the number of tasks this model predicts
    feature_ids: list of str
      the unique identifiers for the features this model generates predictions
      based on.  These strings must match the feature IDs in the OntologyNodes.
      The first element of this list must correspond to the first feature in the
      data, the second element to the second feature, etc.
    root_node: OntologyNode
      the root node of the ontology that defines this models
    mode: str
      the type of model to create, either "regression" or "classification"
    n_classes: int
      for classification models, the number of classes to predict.  This is
      ignored for regression models.
    intermediate_loss_weight: float
      the weight to multiply the loss from intermediate (non-root) categories by
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use for normalization
    """
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


class OntologyNode(object):
  """An OntologyNode represents a category within an ontology."""

  def __init__(self,
               node_id=None,
               n_outputs=10,
               feature_ids=None,
               children=None,
               name=None):
    """Create a OntologyNode representing a category in an ontology.

    Parameters
    ----------
    node_id: str
      a unique identifier for this category.  If this is omitted, an identifier
      is generated automatically.
    n_outputs: int
      the number of output values the corresponding layer of the OntologyModel
      should produce
    feature_ids: list of str
      the unique IDs of all features that belong to this category (not including
      ones that belong to child nodes)
    children: list of OntologyNode
      the list of nodes defining subcategories
    name: str
      a descriptive name for this category.  Ir this is omitted, the name is set
      to the ID.
    """
    self.id = node_id
    self.n_outputs = n_outputs
    self.feature_ids = (feature_ids if feature_ids is not None else [])
    self.children = (children if children is not None else [])
    self.name = (name if name is not None else id)


def create_gene_ontology(feature_mapping,
                         outputs_per_feature=0.3,
                         min_outputs=20,
                         min_node_features=6,
                         omit_redundant_nodes=True,
                         ontology_file=None):
  """Create a tree of OntologyNodes describing the Gene Ontology classification.

  See http://geneontology.org/ for details about the Gene Ontology classification.

  Parameters
  ----------
  feature_mapping: dict
    defines the mapping of features to GO categories.  Each key should be a
    feature ID.  The corresponding value should be a list of strings, giving the
    unique identifiers of all GO categories that feature belongs to.
  outputs_per_feature: float
    the number of outputs for each node is set to this value times the total
    number of features the node contains (including all subnodes)
  min_outputs: int
    the minimum number of outputs for any node
  min_node_features: int
    the minimum number of features corresponding to a node (including all its
    subnodes).  If a category has fewer features than this, no node is create
    for it.  Instead, its features are added directly to its parent node.
  omit_redundant_nodes: bool
    if True, a node will be omitted if it has only one child node and does not
    directly directly correspond to any features
  ontology_file: str
    the path to a Gene Ontology OBO file defining the ontology.  If this is
    omitted, the most recent version of the ontology is downloaded from the GO
    website.
  """
  # If necessary, download the file defining the ontology.

  if ontology_file is None:
    ontology_file = os.path.join(get_data_dir(), 'go-basic.obo')
    if not os.path.isfile(ontology_file):
      download_url('http://purl.obolibrary.org/obo/go/go-basic.obo')

  # Parse the ontology definition and create a list of terms.

  terms = []
  term = None
  with open(ontology_file) as input:
    for line in input:
      if line.startswith('[Term]'):
        if term is not None:
          terms.append(term)
        term = {'parents': []}
      elif line.startswith('[Typedef]'):
        if term is not None:
          terms.append(term)
        term = None
      elif line.startswith('id:') and term is not None:
        term['id'] = line.split()[1]
      elif line.startswith('name:') and term is not None:
        term['name'] = line[5:].strip()
      elif line.startswith('is_a:') and term is not None:
        term['parents'].append(line.split()[1])
      elif line.startswith('is_obsolete:'):
        if line.split()[1] == 'true':
          term = None
  if term is not None:
    terms.append(term)

  # Create OntologyNode objects for all the terms.

  nodes = {}
  for term in terms:
    nodes[term['id']] = OntologyNode(term['id'], 0, name=term['name'])

  # Assign parent-child relationships between nodes, and identify root nodes.

  roots = []
  for term in terms:
    node = nodes[term['id']]
    for parent in term['parents']:
      nodes[parent].children.append(node)
    if len(term['parents']) == 0:
      roots.append(node)

  # Create a single root node that combines the three GO roots.

  root = OntologyNode('GO', 0, name='Gene Ontology Root Node', children=roots)

  # Assign features to nodes.

  for feature_id in feature_mapping:
    for node_id in feature_mapping[feature_id]:
      nodes[node_id].feature_ids.append(feature_id)

  # Count the number of features within each node.  Eliminate nodes with too few
  # features and set the number of outputs for each one.

  def count_features(node):
    self_features = set(node.feature_ids)
    all_features = set(node.feature_ids)
    for i, child in enumerate(node.children[:]):
      child_features = count_features(child)
      all_features.update(child_features)
      if len(child_features) < min_node_features:
        node.children.remove(child)
        self_features.update(child.feature_ids)
    if omit_redundant_nodes and len(
        node.children) == 1 and len(self_features) == 0:
      self_features = node.children[0].feature_ids
      node.children = node.children[0].children
    n_features = len(self_features)
    if n_features > len(node.feature_ids):
      node.feature_ids = list(self_features)
    node.n_outputs = max(min_outputs,
                         math.ceil(outputs_per_feature * n_features))
    return all_features

  count_features(root)
  return root
