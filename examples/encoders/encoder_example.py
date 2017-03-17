from rdkit import Chem

import tensorflow as tf

from data import DiskDataset
from deepchem.data import NumpyDataset
from deepchem.models.autoencoder_models.autoencoder import TensorflowMoleculeEncoder, TensorflowMoleculeDecoder
from deepchem.feat.one_hot import OneHotFeaturizer, zinc_charset
from models.tf_new_models.tensor_graph import TensorGraph, Conv1DLayer, Flatten, Dense, CombineMeanStd, GRU, \
  TimeSeriesDense, Input, Repeat
from nn.copy import Layer


def main():
  tf_enc = TensorflowMoleculeEncoder.zinc_encoder()

  smiles = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)N1CN(C(C)=O)[C@@H](O)[C@@H]1O"]
  mols = [Chem.MolFromSmiles(x) for x in smiles]
  featurizer = OneHotFeaturizer(zinc_charset)
  features = featurizer.featurize(mols)

  dataset = NumpyDataset(features, features)
  prediction = tf_enc.predict_on_batch(dataset.X)

  tf_de = TensorflowMoleculeDecoder.zinc_decoder()
  one_hot_decoded = tf_de.predict_on_batch(prediction)
  print(featurizer.untransform(one_hot_decoded))


class LossLayer(Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def set_parents(self, parents):
    z_mean, z_std, input_layer, decoded = parents[0], parents[1], parents[
        2], parents[3]
    latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean.out_tensor) + \
                                      tf.square(z_std.out_tensor) - \
                                      tf.log(tf.square(z_std.out_tensor)) - 1, 1)
    generation_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=decoded.out_tensor, labels=input_layer.out_tensor))
    self.out_tensor = tf.reduce_mean(generation_loss + latent_loss)


def main2():
  dense_hidden_size = 435
  latent_vector_size = 292
  gru_hidden = 100
  gru_channels = 501
  batch_size = 200
  charset_size = 47
  max_smile_size = 120

  # import json
  #
  # smiles = json.load(open("/home/leswing/Documents/keras-molecules/out.json"))
  # mols = []
  # for smile in smiles:
  #   smile = "".join(smile)
  #   try:
  #     mol = Chem.MolFromSmiles(smile)
  #     if mol is None:
  #       continue
  #     mols.append(mol)
  #   except Exception as e:
  #     print(e, smile)
  #
  # featurizer = OneHotFeaturizer()
  # features = featurizer.featurize(mols)
  #
  # dataset = NumpyDataset(features, features)
  dataset = DiskDataset(data_dir='/tmp/tmpza_f9bd_')
  print(dataset.X.shape)

  graph_model = TensorGraph(
      data_dir="/home/leswing/Documents/graph_model/graph_model")
  input_layer = Input(
      (batch_size, max_smile_size, charset_size), name="features")
  conv1 = Conv1DLayer(width=9, out_channels=9, name="conv1")
  conv2 = Conv1DLayer(width=9, out_channels=9, name="conv2")
  conv3 = Conv1DLayer(width=10, out_channels=11, name="conv3")
  flatten = Flatten(name="flatten")
  dense1 = Dense(out_channels=dense_hidden_size, name="dense1")
  z_mean = Dense(out_channels=latent_vector_size, name="z_mean")
  z_std = Dense(out_channels=latent_vector_size, name="z_std")
  latent = CombineMeanStd(name="latent")
  dense2 = Dense(out_channels=latent_vector_size, name="dense2")
  repeat1 = Repeat(n_times=max_smile_size)
  gru1 = GRU(n_hidden=gru_hidden,
             out_channels=gru_channels,
             batch_size=batch_size,
             name="GRU1")
  gru2 = GRU(n_hidden=gru_hidden,
             out_channels=gru_channels,
             batch_size=batch_size,
             name="GRU2")
  gru3 = GRU(n_hidden=gru_hidden,
             out_channels=gru_channels,
             batch_size=batch_size,
             name="GRU3")
  decoded = TimeSeriesDense(out_channels=charset_size, name="decoded")

  graph_model.add_layer(input_layer, parents=list())
  graph_model.add_layer(conv1, parents=[input_layer])
  graph_model.add_layer(conv2, parents=[conv1])
  graph_model.add_layer(conv3, parents=[conv2])
  graph_model.add_layer(flatten, parents=[conv3])
  graph_model.add_layer(dense1, parents=[flatten])
  graph_model.add_layer(z_mean, parents=[dense1])
  graph_model.add_layer(z_std, parents=[dense1])
  graph_model.add_layer(latent, parents=[z_mean, z_std])
  graph_model.add_layer(dense2, parents=[latent])
  graph_model.add_layer(repeat1, parents=[dense2])
  graph_model.add_layer(gru1, parents=[repeat1])
  graph_model.add_layer(gru2, parents=[gru1])
  graph_model.add_layer(gru3, parents=[gru2])
  graph_model.add_layer(decoded, parents=[gru3])

  loss_layer = graph_model.add_layer(
      LossLayer(name="loss"), parents=[z_mean, z_std, input_layer, decoded])

  labels = graph_model.add_layer(
      Input((batch_size, charset_size * max_smile_size), name="labels"),
      parents=[])
  graph_model.features = input_layer.out_tensor
  graph_model.labels = labels.out_tensor
  graph_model.outputs = decoded.out_tensor
  graph_model.loss = loss_layer.out_tensor

  # end graph default
  graph_model.fit(dataset, nb_epoch=1000, batch_size=batch_size)


if __name__ == "__main__":
  # main()
  main2()
