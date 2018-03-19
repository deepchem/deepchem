import deepchem as dc
import tensorflow as tensorflow
from deepchem.models.tensorgraph.layers import Layer, Input, Dense, Hingeloss, Label, Feature
from deepchem.models.tensorgraph.layers import ReduceMean, InputFifoQueue, ReLU, Add
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class Scscore(TensorGraph):

  tg = dc.models.TensorGraph(
      tensorboard=True, model_dir='/tmp/scscore', use_queue=True)

  reactant_features = Feature()
  product_features = Feature()

  dense_reactant_1 = Dense(
      out_channels=300, in_layers=[reactant_features], activation_fn=tf.nn.relu)
  dense_product_1 = dense_reactant_1.shared(in_layers=[product_features])
  dense_reactant_2 = Dense(
      out_channels=300, in_layers=[dense_reactant_1], activation_fn=tf.nn.relu)
  dense_product_2 = dense_reactant_2.shared(in_layers=[dense_product_1])
  dense_reactant_3 = Dense(
      out_channels=300, in_layers=[dense_reactant_2], activation_fn=tf.nn.relu)
  dense_product_3 = dense_reactant_3.shared(in_layers=[dense_product_2])
  dense_reactant_4 = Dense(
      out_channels=300, in_layers=[dense_reactant_3], activation_fn=tf.nn.relu)
  dense_product_4 = dense_reactant_4.shared(in_layers=[dense_product_3])
  dense_reactant_5 = Dense(
      out_channels=300, in_layers=[dense_reactant_4], activation_fn=tf.nn.relu)
  dense_product_5 = dense_reactant_5.shared(in_layers=[dense_product_4])

  output_reactant = Sigmoid(in_layers=[dense_reactant_5])
  output_product = Sigmoid(in_layers=[dense_product_5])
  output = tf.subtract(output_product, output_reactant)
  tg.add_output(output)

  label = Label(shape=(None, 1))

  loss = Hingeloss(in_layers=[output_product, output_reactant])
  tg.set_loss(loss)
