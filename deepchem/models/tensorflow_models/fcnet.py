"""TensorFlow implementation of the models from the ICML-2015 paper.


hyperparam_dict = {
    "single": Hyperparams(num_layers=1,
                          num_hidden=1200,
                          node_depth=1,
                          nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                          weight_init=GaussianWeightInit(0.01),
                          bias_init=ConstantBiasInit(0.5),
                          dropout=1.),
    "deep": Hyperparams(num_layers=4,
                        num_hidden=1000,
                        node_depth=1,
                        nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                        weight_init=GaussianWeightInit(0.01),
                        bias_init=ConstantBiasInit(0.5),
                        dropout=1.),
    "deepaux": Hyperparams(num_layers=4,
                        num_hidden=1000,
                        auxiliary_softmax_layers=[0, 1, 2],
                        auxiliary_softmax_weight=0.3,
                        node_depth=1,
                        nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                        weight_init=GaussianWeightInit(0.01),
                        bias_init=ConstantBiasInit(0.5),
                        dropout=1.),
    "py": Hyperparams(num_layers=2,
                      num_hidden=[2000, 100],
                      node_depth=1,
                      nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                      weight_init=[GaussianWeightInit(0.01),
                                   GaussianWeightInit(0.04)],
                      bias_init=[ConstantBiasInit(0.5),
                                 ConstantBiasInit(3.0)],
                      dropout=1.),
    "pydrop1": Hyperparams(num_layers=2,
                           num_hidden=[2000, 100],
                           node_depth=1,
                           nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                           weight_init=[GaussianWeightInit(0.01),
                                        GaussianWeightInit(0.04)],
                           bias_init=[ConstantBiasInit(0.5),
                                      ConstantBiasInit(3.0)],
                           dropout=[0.75, 1.]),
    "pydrop2": Hyperparams(num_layers=2,
                           num_hidden=[2000, 100],
                           node_depth=1,
                           nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                           weight_init=[GaussianWeightInit(0.01),
                                        GaussianWeightInit(0.04)],
                           bias_init=[ConstantBiasInit(0.5),
                                      ConstantBiasInit(3.0)],
                           dropout=[0.75, 0.75])}
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#!/usr/bin/python
#
# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import logging

from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import model_ops
from deepchem.utils.evaluate import to_one_hot

class TensorflowMultiTaskClassifier(TensorflowClassifier):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x num_features.
    """
    with self.graph.as_default():
      with tf.name_scope(self.placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[self.model_params["batch_size"],
                   self.model_params["num_features"]],
            name='mol_features')

      layer_sizes = self.model_params["layer_sizes"]
      weight_init_stddevs = self.model_params["weight_init_stddevs"]
      bias_init_consts = self.model_params["bias_init_consts"]
      dropouts = self.model_params["dropouts"]
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
          }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      num_layers = lengths_set.pop()
      assert num_layers > 0, 'Must have some layers defined.'

      prev_layer = self.mol_features
      prev_layer_size = self.model_params["num_features"]
      for i in xrange(num_layers):
        layer = tf.nn.relu(model_ops.FullyConnectedLayer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.Dropout(layer, dropouts[i])
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      self.output = model_ops.MultitaskLogits(
          layer, self.model_params["num_classification_tasks"])

  # TODO(rbharath): Copying this out for now. Ensure this isn't harmful
  #def add_labels_and_weights(self):
  #  """Parse Label protos and create tensors for labels and weights.

  #  This method creates the following Placeholders in the graph:
  #    labels: Tensor with shape batch_size x num_tasks containing serialized
  #      Label protos.
  #  """
  #  config = self.config
  #  with tf.name_scope(self.placeholder_scope):
  #    labels = tf.placeholder(
  #        tf.string,
  #        shape=[config.batch_size, config.num_classification_tasks],
  #        name='labels')
  #  self.labels = label_ops.MultitaskLabelClasses(labels, config.num_classes)
  #  self.weights = label_ops.MultitaskLabelWeights(labels)

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, num_features)
      y_b: np.ndarray of shape (batch_size, num_tasks)
      w_b: np.ndarray of shape (batch_size, num_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """ 
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in xrange(self.num_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = to_one_hot(y_b[:, task])
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(to_one_hot(
            np.zeros((self.model_params["batch_size"],))))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.model_params["batch_size"],)) 
    orig_dict["valid"] = np.ones((self.model_params["batch_size"],), dtype=bool)
    return self._get_feed_dict(orig_dict)

  # TODO(rbharath): This explicit manipulation of scopes is ugly. Is there a
  # better design here?
  def _get_feed_dict(self, named_values):
    feed_dict = {}
    for name, value in named_values.iteritems():
      feed_dict['{}/{}:0'.format(self.placeholder_root, name)] = value
    return feed_dict

  def ReadInput(self, input_pattern, input_data_types=None):
    """Read input data and return a generator for minibatches.

    Args:
      input_pattern: Input file pattern.
      input_data_types: List of legacy_types_pb2 constants matching the
          number of and data types present in the sstables. If not specified,
          defaults to full ICML 259-task types, but can be specified
          for unittests or other datasets with consistent types.

    Returns:
      A generator that yields a dict for feeding a single batch to Placeholders
      in the graph.

    Raises:
      AssertionError: If no default session is available.
    """
    if model_ops.IsTraining():
      randomize = True
      num_iterations = None
    else:
      randomize = False
      num_iterations = 1

    num_tasks = self.model_params["num_classification_tasks"]
    tasks_in_input = self.model_params["tasks_in_input"]
    if input_data_types is None:
      input_data_types = ([legacy_types_pb2.DF_FLOAT] +
                          [legacy_types_pb2.DF_LABEL_PROTO] * tasks_in_input)
    features, labels = input_ops.InputExampleInputReader(
        input_pattern=input_pattern,
        batch_size=self.model_params["batch_size"],
        num_tasks=num_tasks,
        input_data_types=input_data_types,
        num_features=self.model_params["num_features"],
        randomize=randomize,
        shuffling=randomize,
        num_iterations=num_iterations)

    return self._ReadInputGenerator(features, labels[:, :num_tasks])

  def _GetFeedDict(self, named_values):
    feed_dict = {}
    for name, value in named_values.iteritems():
      feed_dict['{}/{}:0'.format(self.placeholder_root, name)] = value

    return feed_dict

  def EvalBatch(self, input_batch):
    """Runs inference on the provided batch of input.

    Args:
      input_batch: iterator of input with len self.model_params["batch_size"].

    Returns:
      Tuple of three numpy arrays with shape num_examples x num_tasks (x ...):
        output: Model predictions.
        labels: True labels. numpy array values are scalars,
            not 1-hot classes vector.
        weights: Example weights.
    """
    output, labels, weights = super(TensorflowMultiTaskDNN, self).EvalBatch(
        input_batch)

    # Converts labels from 1-hot to float.
    labels = labels[:, :, 1]  # Whole batch, all tasks, 1-hot positive index.
    return output, labels, weights

  def BatchInputGenerator(self, serialized_batch):
    """Returns a generator that iterates over the provided batch of input.

    TODO(user): This is similar to input_ops.InputExampleInputReader(),
        but doesn't need to be executed as part of the TensorFlow graph.
        Consider refactoring so these can share code somehow.

    Args:
      serialized_batch: List of tuples: (_, value) where value is
          a serialized InputExample proto. Must have self.model_params["batch_size"]
          length or smaller. If smaller, we'll pad up to batch_size
          and mark the padding as invalid so it's ignored in eval metrics.
    Yields:
      Dict of model inputs for use as a feed_dict.

    Raises:
      ValueError: If the batch is larger than the batch_size.
    """
    if len(serialized_batch) > self.model_params["batch_size"]:
      raise ValueError(
          'serialized_batch length {} must be <= batch_size {}'.format(
              len(serialized_batch), self.model_params["batch_size"]))
    for _ in xrange(self.model_params["batch_size"] - len(serialized_batch)):
      serialized_batch.append((None, ''))

    features = []
    labels = []
    for _, serialized_proto in serialized_batch:
      if serialized_proto:
        input_example = input_example_pb2.InputExample()
        input_example.ParseFromString(serialized_proto)
        features.append([f for f in input_example.endpoint[0].float_value])
        label_protos = [endpoint.label
                        for endpoint in input_example.endpoint[1:]]
        assert len(label_protos) == self.model_params["num_classification_tasks"]
        labels.append([l.SerializeToString() for l in label_protos])
      else:
        # This was a padded value to reach the batch size.
        features.append([0.0 for _ in xrange(self.model_params["num_features"])])
        labels.append(
            ['' for _ in xrange(self.model_params["num_classification_tasks"])])

    valid = np.asarray([(np.sum(f) > 0) for f in features])

    assert len(features) == self.model_params["batch_size"]
    assert len(labels) == self.model_params["batch_size"]
    assert len(valid) == self.model_params["batch_size"]
    yield self._GetFeedDict({
        'mol_features': features,
        'labels': labels,
        'valid': valid
    })

  def _ReadInputGenerator(self, features_tensor, labels_tensor):
    """Generator that constructs feed_dict for minibatches.

    Args:
      features_tensor: Tensor of batch_size x molecule features.
      labels_tensor: Tensor of batch_size x label protos.

    Yields:
      A dict for feeding a single batch to Placeholders in the graph.

    Raises:
      AssertionError: If no default session is available.
    """
    sess = tf.get_default_session()
    if sess is None:
      raise AssertionError('No default session')
    while True:
      try:
        logging.vlog(1, 'Starting session execution to get input data')
        features, labels = sess.run([features_tensor, labels_tensor])
        logging.vlog(1, 'Done with session execution to get input data')
        # TODO(user): check if the below axis=1 needs to change to axis=0,
        # because cl/105081140.
        valid = np.sum(features, axis=1) > 0
        yield self._GetFeedDict({
            'mol_features': features,
            'labels': labels,
            'valid': valid
        })

      except tf.OpError as e:
        # InputExampleInput op raises OpError when it has hit num_iterations
        # or its input file is exhausted. However it may also be raised
        # if the input sstable isn't what we expect.
        if 'Invalid InputExample' in e.message:
          raise e
        else:
          break

  def Run(self, input_data_types=None):
    """Trains the model with specified parameters.

    Args:
      input_data_types: List of legacy_types_pb2 constants or None.
    """
    model_params = model_config.ModelConfig({
        'input_pattern': '',  # Should have %d for fold index substitution.
        'num_classification_tasks': 259,
        'tasks_in_input': 259,  # Dimensionality of sstables
        'max_steps': 50000000,
        'summaries': False,
        'batch_size': 128,
        'learning_rate': 0.0003,
        'num_classes': 2,
        'optimizer': 'sgd',
        'penalty': 0.0,
        'num_features': 1024,
        'layer_sizes': [1200],
        'weight_init_stddevs': [0.01],
        'bias_init_consts': [0.5],
        'dropouts': [0.0],
    })
    #model_params.ReadFromFile(FLAGS.config,
    #                          overwrite='required')

    if FLAGS.replica_id == 0:
      gfile.MakeDirs(FLAGS.logdir)
      #model_params.WriteToFile(os.path.join(FLAGS.logdir, 'config.pbtxt'))

#    model = icml_models.IcmlModel(config,
#                                  train=True,
#                                  logdir=FLAGS.logdir,
#                                  master=FLAGS.master)

    if FLAGS.num_folds is not None and FLAGS.fold is not None:
      folds = kfold_pattern(config.input_pattern, FLAGS.num_folds,
                            FLAGS.fold)
      train_pattern, _ = folds.next()
      train_pattern = ','.join(train_pattern)
    else:
      train_pattern = config.input_pattern

    with model.graph.as_default():
      model.fit(model.read_input(train_pattern,
                                 input_data_types=input_data_types),
                max_steps=config.max_steps,
                summaries=config.summaries,
                replica_id=FLAGS.replica_id,
                ps_tasks=FLAGS.ps_tasks)
