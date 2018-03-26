import numpy as np
import tensorflow as tf
from deepchem.data import NumpyDataset
from deepchem.feat import CircularFingerprint
from deepchem.models.tensorgraph.layers import Dense, HingeLoss, Sigmoid, \
  WeightedError, Dropout
from deepchem.models.tensorgraph.layers import Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class ScScoreModel(TensorGraph):
  """
  https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00622
  Several definitions of molecular complexity exist to facilitate prioritization
  of lead compounds, to identify diversity-inducing and complexifying reactions,
  and to guide retrosynthetic searches. In this work, we focus on synthetic
  complexity and reformalize its definition to correlate with the expected number
  of reaction steps required to produce a target molecule, with implicit knowledge
  about what compounds are reasonable starting materials. We train a neural
  network model on 12 million reactions from the Reaxys database to impose a
  pairwise inequality constraint enforcing the premise of this definition: that on
  average, the products of published chemical reactions should be more
  synthetically complex than their corresponding reactants. The learned metric
  (SCScore) exhibits highly desirable nonlinear behavior, particularly in
  recognizing increases in synthetic complexity throughout a number of linear
  synthetic routes.

  Our model here actually uses hingeloss instead of the shifted relu loss in
  https://github.com/connorcoley/scscore.

  This could cause issues differentiation issues with compounds that are "close"
  to each other in "complexity"

  """

  def __init__(self,
               n_features,
               layer_sizes=[300, 300, 300],
               dropouts=0.0,
               **kwargs):
    """
    Parameters
    ----------
    n_features: int
      number of features per molecule
    layer_sizes: list of int
      size of each hidden layer
    dropouts: int
      droupout to apply to each hidden layer
    kwargs
      This takes all kwards as TensorGraph
    """
    self.n_features = n_features
    self.layer_sizes = layer_sizes
    self.dropout = dropouts
    super(ScScoreModel, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    """
    Building graph structures:
    """
    self.m1_features = Feature(shape=(None, self.n_features))
    self.m2_features = Feature(shape=(None, self.n_features))
    prev_layer1 = self.m1_features
    prev_layer2 = self.m2_features
    for layer_size in self.layer_sizes:
      prev_layer1 = Dense(
          out_channels=layer_size,
          in_layers=[prev_layer1],
          activation_fn=tf.nn.relu)
      prev_layer2 = prev_layer1.shared([prev_layer2])
      if self.dropout > 0.0:
        prev_layer1 = Dropout(self.dropout, in_layers=prev_layer1)
        prev_layer2 = Dropout(self.dropout, in_layers=prev_layer2)

    readout_m1 = Dense(
        out_channels=1, in_layers=[prev_layer1], activation_fn=None)
    readout_m2 = readout_m1.shared([prev_layer2])
    self.add_output(Sigmoid(readout_m1) * 4 + 1)
    self.add_output(Sigmoid(readout_m2) * 4 + 1)

    self.difference = readout_m1 - readout_m2
    label = Label(shape=(None, 1))
    loss = HingeLoss(in_layers=[label, self.difference])
    self.my_task_weights = Weights(shape=(None, 1))
    loss = WeightedError(in_layers=[loss, self.my_task_weights])
    self.set_loss(loss)

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
        feed_dict[self.m1_features] = X_b[:, 0]
        feed_dict[self.m2_features] = X_b[:, 1]
        if y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b
        if w_b is not None and not predict:
          feed_dict[self.my_task_weights] = w_b
        yield feed_dict

  def predict_mols(self, mols):
    featurizer = CircularFingerprint(
        size=self.n_features, radius=2, chiral=True)
    features = np.expand_dims(featurizer.featurize(mols), axis=1)
    features = np.concatenate([features, features], axis=1)
    ds = NumpyDataset(features, None, None, None)
    return self.predict(ds)[0][:, 0]

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    tensors = {}
    for layer, column in zip([self.m1_features, self.m2_features],
                             feature_columns):
      tensors[layer] = tf.feature_column.input_layer(features, [column])
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
          features, [weight_column])
    if labels is not None:
      tensors[self.labels[0]] = tf.cast(labels, tf.int32)
    return tensors
