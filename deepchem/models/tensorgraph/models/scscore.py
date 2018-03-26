import deepchem as dc
import tensorflow as tensorflow
from deepchem.models.tensorgraph.layers import Layer, Input, Dense, Hingeloss, Label, Feature
from deepchem.models.tensorgraph.layers import ReduceMean, InputFifoQueue, ReLU, Add
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class SCScore(Tensorgraph):

  def __init__(self,
               FP_len=1024,
               FP_rad=2,
               score_scale=5.0,
               offset_loss=0.25,
               **kwargs):

    self.FP_len = FP_len
    self.FP_rad = FP_rad
    self.score_scale = score_scale
    self.offset_loss = offset_loss
    super(SCScore, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):

    self.reactant_features = Feature(shape=())
    self.product_features = Feature(shape=())

    in_layer_reactant = self.reactant_features
    in_layer_product = self.product_features

  dense_reactant_1 = Dense(
      out_channels=300, in_layers=[in_layer_reactant], activation_fn=tf.nn.relu)
  dense_product_1 = dense_reactant_1.shared(in_layers=[in_layer_product])
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
      out_channels=1, in_layers=[dense_reactant_4], activation_fn=tf.nn.relu)
  dense_product_5 = dense_reactant_5.shared(in_layers=[dense_product_4])

  output_score_reactant = Sigmoid(in_layers=[dense_reactant_5])
  output_score_product = Sigmoid(in_layers=[dense_product_5])

  scaled_score_reactant = 1.0 + (self.score_scale - 1.0) * output_score_reactant
  scaled_score_product = 1.0 + (self.score_scale - 1.0) * output_score_product

  output = scaled_score_product - scaled_score_reactant
  self.add_output(output)

  label = Label(shape=(None, 1))
  self.my_labels.append(label)

  modified_output = output + 0.75

  loss = Hingeloss(in_layers=[self.my_labels, modified_output])
  self.set_loss(loss)
