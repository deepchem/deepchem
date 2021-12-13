import torch
import torch.nn as nn
import numpy as np
from deepchem.models.torch_models import layers
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss
from typing import Any


class MAT(nn.Module):
  '''An internal TorchModel class.

  In this class, we define the various layers and establish a sequential model for the Molecular Attention Transformer.
  We also define the forward call of this model in the forward function.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> import deepchem as dc
  >>> import pandas as pd
  >>> import numpy as np
  >>> smiles = ['CC', 'CCC',  'CCCC', 'CCCCC', 'CCCCCCC']
  >>> vals = [1.35, 6.72, 5.67, 1.23, 1.76]
  >>> df = pd.DataFrame(list(zip(smiles, vals)), columns = ['smiles', 'y'])
  >>> loader = dc.data.CSVLoader(tasks=['y'], feature_field='smiles', featurizer=dc.feat.MATFeaturizer())
  >>> df.to_csv('test.csv')
  >>> dataset = loader.create_dataset('test.csv')
  >>> model = dc.models.torch_models.MAT()
  >>> # To simulate input data, we will generate matrices for a single molecule.
  >>> vals = dataset.X[0]
  >>> node = vals.node_features
  >>> adj = vals.adjacency_matrix
  >>> dist = vals.distance_matrix
  >>> # We will now utilize a helper function defined in MATModel to get our matrices ready, and convert them into a batch consisting of a single molecule.
  >>> helper = dc.models.torch_models.MATModel()
  >>> node_features = helper.pad_sequence(torch.tensor(node).unsqueeze(0).float())
  >>> adjacency = helper.pad_sequence(torch.tensor(adj).unsqueeze(0).float())
  >>> distance = helper.pad_sequence(torch.tensor(dist).unsqueeze(0).float())
  >>> inputs = [node_features, adjacency, distance]
  >>> inputs = [x.astype(np.float32) if x.dtype == np.float64 else x for x in inputs]
  >>> # Get the forward call of the model for this batch.
  >>> output = model(inputs)
  '''

  def __init__(self,
               dist_kernel: str = 'softmax',
               n_encoders=8,
               lambda_attention: float = 0.33,
               lambda_distance: float = 0.33,
               h: int = 16,
               sa_hsize: int = 1024,
               sa_dropout_p: float = 0.0,
               output_bias: bool = True,
               d_input: int = 1024,
               d_hidden: int = 1024,
               d_output: int = 1024,
               activation: str = 'leakyrelu',
               n_layers: int = 1,
               ff_dropout_p: float = 0.0,
               encoder_hsize: int = 1024,
               encoder_dropout_p: float = 0.0,
               embed_input_hsize: int = 36,
               embed_dropout_p: float = 0.0,
               gen_aggregation_type: str = 'mean',
               gen_dropout_p: float = 0.0,
               gen_n_layers: int = 1,
               gen_attn_hidden: int = 128,
               gen_attn_out: int = 4,
               gen_d_output: int = 1,
               **kwargs):
    '''
    Initialization for the internal MAT class.

    Parameters
    ----------
    dist_kernel: str
        Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential, for the self-attention layer.
    n_encoders: int
        Number of encoder layers in the encoder block.
    lambda_attention: float
        Constant to be multiplied with the attention matrix in the self-attention layer.
    lambda_distance: float
        Constant to be multiplied with the distance matrix in the self-attention layer.
    h: int
        Number of attention heads for the self-attention layer.
    sa_hsize: int
        Size of dense layer in the self-attention layer.
    sa_dropout_p: float
        Dropout probability for the self-attention layer.
    output_bias: bool
        If True, dense layers will use bias vectors in the self-attention layer.
    d_input: int
        Size of input layer in the feed-forward layer.
    d_hidden: int
        Size of hidden layer in the feed-forward layer. Will also be used as d_output for the MATEmbedding layer.
    d_output: int
        Size of output layer in the feed-forward layer.
    activation: str
        Activation function to be used in the feed-forward layer.
        Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
        'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
    n_layers: int
        Number of layers in the feed-forward layer.
    ff_dropout_p: float
        Dropout probability in the feeed-forward layer.
    encoder_hsize: int
        Size of Dense layer for the encoder itself.
    encoder_dropout_p: float
        Dropout probability for connections in the encoder layer.
    embed_input_hsize: int
        Size of input layer for the MATEmbedding layer.
    embed_dropout_p: float
        Dropout probability for the MATEmbedding layer.
    gen_aggregation_type: str
        Type of aggregation to be used. Can be 'grover', 'mean' or 'contextual'.
    gen_dropout_p: float
        Dropout probability for the MATGenerator layer.
    gen_n_layers: int
        Number of layers in MATGenerator.
    gen_attn_hidden: int
        Size of hidden attention layer in the MATGenerator layer.
    gen_attn_out: int
        Size of output attention layer in the MATGenerator layer.
    gen_d_output: int
        Size of output layer in the MATGenerator layer.
    '''

    super(MAT, self).__init__()

    self.embedding = layers.MATEmbedding(
        d_input=embed_input_hsize, d_output=d_hidden, dropout_p=embed_dropout_p)

    self.encoder = nn.ModuleList([
        layers.MATEncoderLayer(
            dist_kernel=dist_kernel,
            lambda_attention=lambda_attention,
            lambda_distance=lambda_distance,
            h=h,
            sa_hsize=sa_hsize,
            sa_dropout_p=sa_dropout_p,
            output_bias=output_bias,
            d_input=d_input,
            d_hidden=d_hidden,
            d_output=d_output,
            activation=activation,
            n_layers=n_layers,
            ff_dropout_p=ff_dropout_p,
            encoder_hsize=encoder_hsize,
            encoder_dropout_p=encoder_dropout_p) for _ in range(n_encoders)
    ])

    self.generator = layers.MATGenerator(
        hsize=d_input,
        aggregation_type=gen_aggregation_type,
        d_output=gen_d_output,
        n_layers=gen_n_layers,
        dropout_p=gen_dropout_p,
        attn_hidden=gen_attn_hidden,
        attn_out=gen_attn_out)

  def forward(self, data: np.ndarray, **kwargs):
    node_features = torch.tensor(data[0]).float()
    adjacency_matrix = torch.tensor(data[1]).float()
    distance_matrix = torch.tensor(data[2]).float()

    mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    output = self.embedding(node_features)

    for layer in self.encoder:
      output = layer(output, mask, adjacency_matrix, distance_matrix)
    output = self.generator(output, mask)
    return output


class MATModel(TorchModel):
  """Molecular Attention Transformer.

  This class implements the Molecular Attention Transformer [1]_.
  The MATFeaturizer (deepchem.feat.MATFeaturizer) is intended to work with this class.
  The model takes a batch of MATEncodings (from MATFeaturizer) as input, and returns an array of size Nx1, where N is the number of molecules in the batch.
  Each molecule is broken down into its Node Features matrix, adjacency matrix and distance matrix.
  A mask tensor is calculated for the batch. All of this goes as input to the MATEmbedding, MATEncoder and MATGenerator layers, which are defined in deepchem.models.torch_models.layers.py

  Currently, MATModel is intended to be a regression model for the freesolv dataset.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> import deepchem as dc
  >>> import pandas as pd
  >>> smiles = ['CC', 'CCC',  'CCCC', 'CCCCC', 'CCCCCCC']
  >>> vals = [1.35, 6.72, 5.67, 1.23, 1.76]
  >>> df = pd.DataFrame(list(zip(smiles, vals)), columns = ['smiles', 'y'])
  >>> loader = dc.data.CSVLoader(tasks=['y'], feature_field='smiles', featurizer=dc.feat.MATFeaturizer())
  >>> df.to_csv('test.csv')
  >>> dataset = loader.create_dataset('test.csv')
  >>> model = dc.models.torch_models.MATModel(batch_size = 2)
  >>> out = model.fit(dataset, nb_epoch = 1)
  """

  def __init__(self,
               dist_kernel: str = 'softmax',
               n_encoders=8,
               lambda_attention: float = 0.33,
               lambda_distance: float = 0.33,
               h: int = 16,
               sa_hsize: int = 1024,
               sa_dropout_p: float = 0.0,
               output_bias: bool = True,
               d_input: int = 1024,
               d_hidden: int = 1024,
               d_output: int = 1024,
               activation: str = 'leakyrelu',
               n_layers: int = 1,
               ff_dropout_p: float = 0.0,
               encoder_hsize: int = 1024,
               encoder_dropout_p: float = 0.0,
               embed_input_hsize: int = 36,
               embed_dropout_p: float = 0.0,
               gen_aggregation_type: str = 'mean',
               gen_dropout_p: float = 0.0,
               gen_n_layers: int = 1,
               gen_attn_hidden: int = 128,
               gen_attn_out: int = 4,
               gen_d_output: int = 1,
               **kwargs):
    """The wrapper class for the Molecular Attention Transformer.

    Since we are using a custom data class as input (MATEncoding), we have overriden the default_generator function from DiskDataset and customized it to work with a batch of MATEncoding classes.

    Parameters
    ----------
    dist_kernel: str
        Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential, for the self-attention layer.
    n_encoders: int
        Number of encoder layers in the encoder block.
    lambda_attention: float
        Constant to be multiplied with the attention matrix in the self-attention layer.
    lambda_distance: float
        Constant to be multiplied with the distance matrix in the self-attention layer.
    h: int
        Number of attention heads for the self-attention layer.
    sa_hsize: int
        Size of dense layer in the self-attention layer.
    sa_dropout_p: float
        Dropout probability for the self-attention layer.
    output_bias: bool
        If True, dense layers will use bias vectors in the self-attention layer.
    d_input: int
        Size of input layer in the feed-forward layer.
    d_hidden: int
        Size of hidden layer in the feed-forward layer. Will also be used as d_output for the MATEmbedding layer.
    d_output: int
        Size of output layer in the feed-forward layer.
    activation: str
        Activation function to be used in the feed-forward layer.
        Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
        'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
    n_layers: int
        Number of layers in the feed-forward layer.
    ff_dropout_p: float
        Dropout probability in the feeed-forward layer.
    encoder_hsize: int
        Size of Dense layer for the encoder itself.
    encoder_dropout_p: float
        Dropout probability for connections in the encoder layer.
    embed_input_hsize: int
        Size of input layer for the MATEmbedding layer.
    embed_dropout_p: float
        Dropout probability for the MATEmbedding layer.
    gen_aggregation_type: str
        Type of aggregation to be used. Can be 'grover', 'mean' or 'contextual'.
    gen_dropout_p: float
        Dropout probability for the MATGenerator layer.
    gen_n_layers: int
        Number of layers in MATGenerator.
    gen_attn_hidden: int
        Size of hidden attention layer in the MATGenerator layer.
    gen_attn_out: int
        Size of output attention layer in the MATGenerator layer.
    gen_d_output: int
        Size of output layer in the MATGenerator layer.
    """
    model = MAT(
        dist_kernel=dist_kernel,
        n_encoders=n_encoders,
        lambda_attention=lambda_attention,
        lambda_distance=lambda_distance,
        h=h,
        sa_hsize=sa_hsize,
        sa_dropout_p=sa_dropout_p,
        output_bias=output_bias,
        d_input=d_input,
        d_hidden=d_hidden,
        d_output=d_output,
        activation=activation,
        n_layers=n_layers,
        ff_dropout_p=ff_dropout_p,
        encoder_hsize=encoder_hsize,
        encoder_dropout_p=encoder_dropout_p,
        embed_input_hsize=embed_input_hsize,
        embed_dropout_p=embed_dropout_p,
        gen_aggregation_type=gen_aggregation_type,
        gen_dropout_p=gen_dropout_p,
        gen_n_layers=gen_n_layers,
        gen_attn_hidden=gen_attn_hidden,
        gen_attn_out=gen_attn_out,
        gen_d_output=gen_d_output)

    loss = L2Loss()
    output_types = ['prediction']
    super(MATModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

  def pad_array(self, array: np.ndarray, shape: Any) -> np.ndarray:
    """
    Pads an array to the desired shape.

    Parameters
    ----------
    array: np.ndarray
    Array to be padded.
    shape: int or Tuple
    Shape the array is padded to.

    Returns
    ----------
    array: np.ndarray
    Array padded to input shape.
    """
    result = np.zeros(shape=shape)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = array
    return result

  def pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
    """
    Pads a given sequence using the pad_array function.

    Parameters
    ----------
    sequence: np.ndarray
    Arrays in this sequence are padded to the largest shape in the sequence.

    Returns
    ----------
    array: np.ndarray
    Sequence with padded arrays.
    """
    shapes = np.stack([np.array(t.shape) for t in sequence])
    max_shape = tuple(np.max(shapes, axis=0))
    return np.stack([self.pad_array(t, shape=max_shape) for t in sequence])

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True,
                        **kwargs):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        node_features = self.pad_sequence(
            [torch.tensor(data.node_features).float() for data in X_b])
        adjacency_matrix = self.pad_sequence(
            [torch.tensor(data.adjacency_matrix).float() for data in X_b])
        distance_matrix = self.pad_sequence(
            [torch.tensor(data.distance_matrix).float() for data in X_b])

        inputs = [node_features, adjacency_matrix, distance_matrix]
        yield (inputs, [y_b], [w_b])
