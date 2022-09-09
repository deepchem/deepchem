import numpy as np

import torch
import torch.nn as nn

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models import layers
from deepchem.models.torch_models import TorchModel

from torch_geometric.data import Data, Batch

from deepchem.feat import GraphData
from deepchem.data import Dataset
from typing import Union, List, Sequence, Optional, Iterable, Tuple


class _ModData(Data):
  """
  Modified version of Data class of pytorch-geometric to enable batching process to
  custom increment values in certain keys.
  """

  def __init__(self, required_inc, *args, **kwargs):
    """
    Initialize the _ModData class
    """
    super().__init__(*args, **kwargs)
    self.required_inc = required_inc  # required increment

  def __inc__(self, key, value, *args, **kwargs):
    """
    Modified __inc__() to increment 'atom_to_incoming_bonds' and 'mapping' keys
    based given required increment value (example, number of bonds in the molecule)
    """
    if key in ['atom_to_incoming_bonds', 'mapping']:
      return self.required_inc
    else:
      return super().__inc__(key, value, *args, **kwargs)


class _MapperDMPNN:
  """
  This class is a helper class for DMPNNModel class to generate concatenated feature vector and mappings.

  `self.f_ini_atoms_bonds` is the concatenated feature vector which contains
  concatenation of initial atom and bond features.

  `self.atom_to_incoming_bonds` is mapping from atom index to list of indicies of incoming bonds.

  `self.mapping` is the mapping that maps bond index to 'array of indices of the bonds'
  incoming at the initial atom of the bond (excluding the reverse bonds)

  Example,
  Let the diagram given below represent a molecule containing 3 atoms (nodes) and 2 bonds (edges):-

  |   0 --- 1
  |   |
  |   2

  Here, atoms are => A0, A1 and A2 and their respective feature vectors are f0, f1, and f2.

  Let the bonds from atoms 0->1 ('B[01]') and 1->0 ('B[10]') be considered as 2 different bonds.
  Hence, by considering the same for all atoms, the total number of bonds = 4.

  Let:
    B[01] => B0
    B[10] => B1
    B[02] => B2
    B[20] => B3

  Hence repective bond features are fb0, fb1, fb2, and fb3.
  (Note: fb0 = fb1, fb2 = fb3)

  'f_ini_atoms_bonds' is the concatenated feature vector which contains
  concatenation of initial atom and bond features.

                   'B0'
    Example: 'A0' -----> A1 , concat feature = f0 + fb0

  Hence,
                            B0       B1       B2       B3     B(-1)
    f_ini_atoms_bonds = [ f0+fb0 , f1+fb1 , f0+fb2 , f2+fb3 , f(-1) ]

    (Note: f(-1) is a zero array of the same size as other concatenated features.)

  `atom_to_incoming_bonds` is mapping from atom index to list of indicies of incoming bonds.

                   B3          B1
    Example: 'A2' ----> 'A0' <---- 'A1', for A0 => [B1, B3]

  Hence,
                                 A0        A1        A2
    atom_to_incoming_bonds = [ [B1,B3] [B0,B(-1)] [B2,B(-1)] ]

    (Note: Here, maximum number of incoming bonds is 2. So, -1 index is added to all those cases
           where number of incoming bonds is less than maximum. In this case, its for A1 and A2.)

  To get mapping, first find indices of the bonds, incoming at the initial atom of the bond.

    Example: for bond B0, B1 and B3 are coming towards atom 0.

  |                    B0             B1
  |                 0 ----> 1  |  0 <---- 1
  |                            |  ^
  |                            |  | B3
  |                            |  2

                                     B0     B1    B2     B3
  mapping (with reverse bonds) = [ [B1,B3] [B0] [B1,B3] [B2] ]

  To get the required mapping, reverse bond indices are replaced with -1
  and extra space in the array elements is filled with -1, to get a uniform array.

  The mapping is also padded with -1 at the end, so that the length of `mapping` is
  equal to the length of `f_ini_atoms_bonds`.

  Hence,
                    B0          B1          B2            B3          B(-1)
    mapping = [ [B(-1),B3] [B(-1),B(-1)] [B1,B(-1)] [B(-1),B(-1)] [B(-1),B(-1)] ]

    OR

    mapping = [[-1, 3], [-1, -1], [1, -1], [-1, -1], [-1, -1]]
  """

  def __init__(self, graph: GraphData):
    """
    Parameters
    ----------
    graph: GraphData
      GraphData object.
    """
    self.num_atoms: int = graph.num_nodes
    self.num_atom_features: int = graph.num_node_features
    self.num_bonds: int = graph.num_edges
    self.num_bond_features: int = graph.num_edge_features
    self.atom_features: np.ndarray = graph.node_features
    self.bond_features: Optional[np.ndarray] = graph.edge_features
    self.bond_index: np.ndarray = graph.edge_index
    self.global_features: np.ndarray = graph.global_features  # type: ignore
    # mypy check is ignored for global_features as it is not a default attribute
    # of GraphData. It is created during runtime using **kwargs.

    # mapping from bond index to the index of the atom (where the bond is coming from)
    self.bond_to_ini_atom: np.ndarray

    # mapping from bond index to concat(in_atom, bond) features
    self.f_ini_atoms_bonds: np.ndarray = np.empty(0)

    # mapping from atom index to list of indicies of incoming bonds
    self.atom_to_incoming_bonds: np.ndarray

    # mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond (excluding the reverse bonds)
    self.mapping: np.ndarray = np.empty(0)

    if self.num_bonds == 0:
      self.bond_to_ini_atom = np.empty(0)
      self.f_ini_atoms_bonds = np.zeros(
          (1, self.num_atom_features + self.num_bond_features))

      self.atom_to_incoming_bonds = np.asarray([[-1]] * self.num_atoms,
                                               dtype=int)
      self.mapping = np.asarray([[-1]], dtype=int)

    else:
      self.bond_to_ini_atom = self.bond_index[0]
      self._get_f_ini_atoms_bonds()  # its zero padded at the end

      self.atom_to_incoming_bonds = self._get_atom_to_incoming_bonds()
      self._generate_mapping()  # its padded with -1 at the end

  @property
  def values(self) -> Sequence[np.ndarray]:
    """
    Returns the required mappings:
    - atom features
    - concat features (atom + bond)
    - atom to incoming bonds mapping
    - mapping
    - global features
    """
    return self.atom_features, self.f_ini_atoms_bonds, self.atom_to_incoming_bonds, self.mapping, self.global_features

  def _get_f_ini_atoms_bonds(self):
    """
    Method to get `self.f_ini_atoms_bonds`
    """
    self.f_ini_atoms_bonds = np.hstack(
        (self.atom_features[self.bond_to_ini_atom], self.bond_features))

    # zero padded at the end
    self.f_ini_atoms_bonds = np.pad(self.f_ini_atoms_bonds, ((0, 1), (0, 0)))

  def _generate_mapping(self):
    """
    Generate mapping, which maps bond index to 'array of indices of the bonds'
    incoming at the initial atom of the bond (reverse bonds are not considered).

    Steps:
    - Get mapping based on `self.atom_to_incoming_bonds` and `self.bond_to_ini_atom`.
    - Replace reverse bond indices with -1.
    - Pad the mapping with -1.
    """

    # get mapping which maps bond index to 'array of indices of the bonds' incoming at the initial atom of the bond
    self.mapping = self.atom_to_incoming_bonds[self.bond_to_ini_atom]
    self._replace_rev_bonds()

    # padded with -1 at the end
    self.mapping = np.pad(self.mapping, ((0, 1), (0, 0)), constant_values=-1)

  def _get_atom_to_incoming_bonds(self) -> np.ndarray:
    """
    Method to get atom_to_incoming_bonds mapping
    """
    # mapping from bond index to the index of the atom (where the bond if going to)
    bond_to_final_atom: np.ndarray = self.bond_index[1]

    # mapping from atom index to list of indicies of incoming bonds
    a2b: List = []
    for i in range(self.num_atoms):
      a2b.append(list(np.where(bond_to_final_atom == i)[0]))

    # get maximum number of incoming bonds
    max_num_bonds: int = max(1,
                             max(len(incoming_bonds) for incoming_bonds in a2b))

    # Make number of incoming bonds equal to maximum number of bonds.
    # This is done by appending -1 to fill remaining space at each atom indices.
    a2b = [
        a2b[a] + [-1] * (max_num_bonds - len(a2b[a]))
        for a in range(self.num_atoms)
    ]

    return np.asarray(a2b, dtype=int)

  def _replace_rev_bonds(self):
    """
    Method to get b2revb and replace the reverse bond indices with -1 in mapping.
    """
    # mapping from bond index to the index of the reverse bond
    b2revb: np.ndarray = np.empty(self.num_bonds, dtype=int)
    for i in range(self.num_bonds):
      if i % 2 == 0:
        b2revb[i] = i + 1
      else:
        b2revb[i] = i - 1

    for count, i in enumerate(b2revb):
      self.mapping[count][np.where(self.mapping[count] == i)] = -1


class DMPNN(nn.Module):
  """
  Directed Message Passing Neural Network

  In this class, we define the various encoder layers and establish a sequential model for the Directed Message Passing Neural Network (D-MPNN) [1]_.
  We also define the forward call of this model in the forward function.

  Example
  -------
  >>> import deepchem as dc
  >>> from torch_geometric.data import Data, Batch
  >>> # Get data
  >>> input_smile = "CC"
  >>> feat = dc.feat.DMPNNFeaturizer(features_generators=['morgan'])
  >>> graph = feat.featurize(input_smile)
  >>> mapper = _MapperDMPNN(graph[0])
  >>> atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = mapper.values
  >>> atom_features = torch.from_numpy(atom_features).float()
  >>> f_ini_atoms_bonds = torch.from_numpy(f_ini_atoms_bonds).float()
  >>> atom_to_incoming_bonds = torch.from_numpy(atom_to_incoming_bonds)
  >>> mapping = torch.from_numpy(mapping)
  >>> global_features = torch.from_numpy(global_features).float()
  >>> data = [Data(atom_features=atom_features,\
  f_ini_atoms_bonds=f_ini_atoms_bonds,\
  atom_to_incoming_bonds=atom_to_incoming_bonds,\
  mapping=mapping, global_features=global_features)]
  >>> # Prepare batch (size 1)
  >>> pyg_batch = Batch()
  >>> pyg_batch = pyg_batch.from_data_list(data)
  >>> # Initialize the model
  >>> model = DMPNN(mode='regression', global_features_size=2048, n_tasks=2)
  >>> # Get the forward call of the model for this batch.
  >>> output = model(pyg_batch)

  References
  ----------
  .. [1] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf
  """

  def __init__(self,
               mode: str = 'regression',
               n_classes: int = 3,
               n_tasks: int = 1,
               global_features_size: int = 0,
               use_default_fdim: bool = True,
               atom_fdim: int = 133,
               bond_fdim: int = 14,
               enc_hidden: int = 300,
               depth: int = 3,
               bias: bool = False,
               enc_activation: str = 'relu',
               enc_dropout_p: float = 0.0,
               aggregation: str = 'mean',
               aggregation_norm: Union[int, float] = 100,
               ffn_hidden: int = 300,
               ffn_activation: str = 'relu',
               ffn_layers: int = 3,
               ffn_dropout_p: float = 0.0,
               ffn_dropout_at_input_no_act: bool = True):
    """
    Initialize the DMPNN class.

    Parameters
    ----------
    mode: str, default 'regression'
      The model type - classification or regression.
    n_classes: int, default 3
      The number of classes to predict (used only in classification mode).
    n_tasks: int, default 1
      The number of tasks.
    global_features_size: int, default 0
      Size of the global features vector, based on the global featurizers used during featurization.
    use_default_fdim: bool
      If `True`, self.atom_fdim and self.bond_fdim are initialized using values from the GraphConvConstants class.
      If `False`, self.atom_fdim and self.bond_fdim are initialized from the values provided.
    atom_fdim: int
      Dimension of atom feature vector.
    bond_fdim: int
      Dimension of bond feature vector.
    enc_hidden: int
      Size of hidden layer in the encoder layer.
    depth: int
      No of message passing steps.
    bias: bool
      If `True`, dense layers will use bias vectors.
    enc_activation: str
      Activation function to be used in the encoder layer.
      Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
      'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
    enc_dropout_p: float
      Dropout probability for the encoder layer.
    aggregation: str
      Aggregation type to be used in the encoder layer.
      Can choose between 'mean', 'sum', and 'norm'.
    aggregation_norm: Union[int, float]
      Value required if `aggregation` type is 'norm'.
    ffn_hidden: int
      Size of hidden layer in the feed-forward network layer.
    ffn_activation: str
      Activation function to be used in feed-forward network layer.
      Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
      'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
    ffn_layers: int
      Number of layers in the feed-forward network layer.
    ffn_dropout_p: float
      Dropout probability for the feed-forward network layer.
    ffn_dropout_at_input_no_act: bool
      If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
    """
    super(DMPNN, self).__init__()
    self.mode: str = mode
    self.n_classes: int = n_classes
    self.n_tasks: int = n_tasks

    # get encoder
    self.encoder: nn.Module = layers.DMPNNEncoderLayer(
        use_default_fdim=use_default_fdim,
        atom_fdim=atom_fdim,
        bond_fdim=bond_fdim,
        d_hidden=enc_hidden,
        depth=depth,
        bias=bias,
        activation=enc_activation,
        dropout_p=enc_dropout_p,
        aggregation=aggregation,
        aggregation_norm=aggregation_norm)

    # get input size for ffn
    ffn_input: int = enc_hidden + global_features_size

    # get output size for ffn
    if self.mode == 'regression':
      ffn_output: int = self.n_tasks
    elif self.mode == 'classification':
      ffn_output = self.n_tasks * self.n_classes

    # get ffn
    self.ffn: nn.Module = layers.PositionwiseFeedForward(
        d_input=ffn_input,
        d_hidden=ffn_hidden,
        d_output=ffn_output,
        activation=ffn_activation,
        n_layers=ffn_layers,
        dropout_p=ffn_dropout_p,
        dropout_at_input_no_act=ffn_dropout_at_input_no_act)

  def forward(self,
              pyg_batch: Batch) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """
    Parameters
    ----------
    data: Batch
      A pytorch-geometric batch containing tensors for:

      - atom_features
      - f_ini_atoms_bonds
      - atom_to_incoming_bonds
      - mapping
      - global_features

      The `molecules_unbatch_key` is also derived from the batch.
      (List containing number of atoms in various molecules of the batch)

    Returns
    -------
    output: Union[torch.Tensor, Sequence[torch.Tensor]]
      Predictions for the graphs
    """
    atom_features: torch.Tensor = pyg_batch['atom_features']
    f_ini_atoms_bonds: torch.Tensor = pyg_batch['f_ini_atoms_bonds']
    atom_to_incoming_bonds: torch.Tensor = pyg_batch['atom_to_incoming_bonds']
    mapping: torch.Tensor = pyg_batch['mapping']
    global_features: torch.Tensor = pyg_batch['global_features']

    # Steps to get `molecules_unbatch_key`:
    # 1. Get the tensor containing the indices of first atoms of each molecule
    # 2. Get the tensor containing number of atoms of each molecule
    #     by taking the difference between consecutive indices.
    # 3. Convert the tensor to a list.
    molecules_unbatch_key: List = torch.diff(
        pyg_batch._slice_dict['atom_features']).tolist()

    # num_molecules x (enc_hidden + global_features_size)
    encodings: torch.Tensor = self.encoder(atom_features, f_ini_atoms_bonds,
                                           atom_to_incoming_bonds, mapping,
                                           global_features,
                                           molecules_unbatch_key)

    # ffn_output (`self.n_tasks` or `self.n_tasks * self.n_classes`)
    output: torch.Tensor = self.ffn(encodings)

    final_output: Union[torch.Tensor, Sequence[torch.Tensor]]

    if self.mode == 'regression':
      final_output = output
    elif self.mode == 'classification':
      if self.n_tasks == 1:
        output = output.view(-1, self.n_classes)
        final_output = nn.functional.softmax(output, dim=1), output
      else:
        output = output.view(-1, self.n_tasks, self.n_classes)
        final_output = nn.functional.softmax(output, dim=2), output

    return final_output


class DMPNNModel(TorchModel):
  """
  Directed Message Passing Neural Network

  This class implements the Directed Message Passing Neural Network (D-MPNN) [1]_.

  The DMPNN model has 2 phases, message-passing phase and read-out phase.

  - The goal of the message-passing phase is to generate 'hidden states of all the atoms in the molecule' using encoders.
  - Next in read-out phase, the features are passed into feed-forward neural network to get the task-based prediction.

  For additional information:

  - `Mapper class <https://github.com/deepchem/deepchem/blob/31676cc2497d5f2de65d648c09fc86191b594501/deepchem/models/torch_models/dmpnn.py#L10-L92>`_
  - `Encoder layer class <https://github.com/deepchem/deepchem/blob/31676cc2497d5f2de65d648c09fc86191b594501/deepchem/models/torch_models/layers.py#L1223-L1374>`_
  - `Feed-Forward class <https://github.com/deepchem/deepchem/blob/31676cc2497d5f2de65d648c09fc86191b594501/deepchem/models/torch_models/layers.py#L689-L700>`_

  Example
  -------
  >>> import deepchem as dc
  >>> import os
  >>> model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  >>> input_file = os.path.join(model_dir, 'tests/assets/freesolv_sample_5.csv')
  >>> loader = dc.data.CSVLoader(tasks=['y'], feature_field='smiles', featurizer=dc.feat.DMPNNFeaturizer())
  >>> dataset = loader.create_dataset(input_file)
  >>> model = DMPNNModel()
  >>> out = model.fit(dataset, nb_epoch=1)

  References
  ----------
  .. [1] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf
  """

  def __init__(self,
               mode: str = 'regression',
               n_classes: int = 3,
               n_tasks: int = 1,
               batch_size: int = 1,
               global_features_size: int = 0,
               use_default_fdim: bool = True,
               atom_fdim: int = 133,
               bond_fdim: int = 14,
               enc_hidden: int = 300,
               depth: int = 3,
               bias: bool = False,
               enc_activation: str = 'relu',
               enc_dropout_p: float = 0.0,
               aggregation: str = 'mean',
               aggregation_norm: Union[int, float] = 100,
               ffn_hidden: int = 300,
               ffn_activation: str = 'relu',
               ffn_layers: int = 3,
               ffn_dropout_p: float = 0.0,
               ffn_dropout_at_input_no_act: bool = True,
               **kwargs):
    """
    Initialize the DMPNNModel class.

    Parameters
    ----------
    mode: str, default 'regression'
      The model type - classification or regression.
    n_classes: int, default 3
      The number of classes to predict (used only in classification mode).
    n_tasks: int, default 1
      The number of tasks.
    batch_size: int, default 1
      The number of datapoints in a batch.
    global_features_size: int, default 0
      Size of the global features vector, based on the global featurizers used during featurization.
    use_default_fdim: bool
      If `True`, self.atom_fdim and self.bond_fdim are initialized using values from the GraphConvConstants class.
      If `False`, self.atom_fdim and self.bond_fdim are initialized from the values provided.
    atom_fdim: int
      Dimension of atom feature vector.
    bond_fdim: int
      Dimension of bond feature vector.
    enc_hidden: int
      Size of hidden layer in the encoder layer.
    depth: int
      No of message passing steps.
    bias: bool
      If `True`, dense layers will use bias vectors.
    enc_activation: str
      Activation function to be used in the encoder layer.
      Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
      'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
    enc_dropout_p: float
      Dropout probability for the encoder layer.
    aggregation: str
      Aggregation type to be used in the encoder layer.
      Can choose between 'mean', 'sum', and 'norm'.
    aggregation_norm: Union[int, float]
      Value required if `aggregation` type is 'norm'.
    ffn_hidden: int
      Size of hidden layer in the feed-forward network layer.
    ffn_activation: str
      Activation function to be used in feed-forward network layer.
      Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
      'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
    ffn_layers: int
      Number of layers in the feed-forward network layer.
    ffn_dropout_p: float
      Dropout probability for the feed-forward network layer.
    ffn_dropout_at_input_no_act: bool
      If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
    kwargs: Dict
      kwargs supported by TorchModel
    """
    model: nn.Module = DMPNN(
        mode=mode,
        n_classes=n_classes,
        n_tasks=n_tasks,
        global_features_size=global_features_size,
        use_default_fdim=use_default_fdim,
        atom_fdim=atom_fdim,
        bond_fdim=bond_fdim,
        enc_hidden=enc_hidden,
        depth=depth,
        bias=bias,
        enc_activation=enc_activation,
        enc_dropout_p=enc_dropout_p,
        aggregation=aggregation,
        aggregation_norm=aggregation_norm,
        ffn_hidden=ffn_hidden,
        ffn_activation=ffn_activation,
        ffn_layers=ffn_layers,
        ffn_dropout_p=ffn_dropout_p,
        ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act)

    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types: List[str] = ['prediction']
    elif mode == 'classification':
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(DMPNNModel, self).__init__(model,
                                     loss=loss,
                                     output_types=output_types,
                                     batch_size=batch_size,
                                     **kwargs)

  def _to_pyg_graph(self, values: Sequence[np.ndarray]) -> _ModData:
    """
    Convert to PyTorch Geometric graph modified data instance

    .. note::
       This method requires PyTorch Geometric to be installed.

    Parameters
    ----------
    values: Sequence[np.ndarray]
      Mappings from ``_MapperDMPNN`` helper class for a molecule

    Returns
    -------
    torch_geometric.data.Data
      Modified Graph data for PyTorch Geometric (``_ModData``)
    """

    # atom feature matrix with shape [number of atoms, number of features]
    atom_features: np.ndarray

    # concatenated feature vector which contains concatenation of initial atom and bond features
    f_ini_atoms_bonds: np.ndarray

    # mapping from atom index to list of indicies of incoming bonds
    atom_to_incoming_bonds: np.ndarray

    # mapping that maps bond index to 'array of indices of the bonds'
    # incoming at the initial atom of the bond (excluding the reverse bonds)
    mapping: np.ndarray

    # array of global molecular features
    global_features: np.ndarray

    atom_features, f_ini_atoms_bonds, atom_to_incoming_bonds, mapping, global_features = values

    t_atom_features: torch.Tensor = torch.from_numpy(atom_features).float().to(
        device=self.device)
    t_f_ini_atoms_bonds: torch.Tensor = torch.from_numpy(
        f_ini_atoms_bonds).float().to(device=self.device)
    t_atom_to_incoming_bonds: torch.Tensor = torch.from_numpy(
        atom_to_incoming_bonds).to(device=self.device)
    t_mapping: torch.Tensor = torch.from_numpy(mapping).to(device=self.device)
    t_global_features: torch.Tensor = torch.from_numpy(
        global_features).float().to(device=self.device)

    return _ModData(required_inc=len(t_f_ini_atoms_bonds),
                    atom_features=t_atom_features,
                    f_ini_atoms_bonds=t_f_ini_atoms_bonds,
                    atom_to_incoming_bonds=t_atom_to_incoming_bonds,
                    mapping=t_mapping,
                    global_features=t_global_features)

  def _prepare_batch(
      self, batch: Tuple[List, List, List]
  ) -> Tuple[Batch, List[torch.Tensor], List[torch.Tensor]]:
    """
    Method to prepare pytorch-geometric batches from inputs.

    Overrides the existing ``_prepare_batch`` method to customize how model batches are
    generated from the inputs.

    .. note::
       This method requires PyTorch Geometric to be installed.

    Parameters
    ----------
    batch: Tuple[List, List, List]
      batch data from ``default_generator``

    Returns
    -------
    Tuple[Batch, List[torch.Tensor], List[torch.Tensor]]
    """
    graphs_list: List
    labels: List
    weights: List

    graphs_list, labels, weights = batch
    pyg_batch: Batch = Batch()
    pyg_batch = pyg_batch.from_data_list(graphs_list)

    _, labels, weights = super(DMPNNModel, self)._prepare_batch(
        ([], labels, weights))
    return pyg_batch, labels, weights

  def default_generator(self,
                        dataset: Dataset,
                        epochs: int = 1,
                        mode: str = 'fit',
                        deterministic: bool = True,
                        pad_batches: bool = False,
                        **kwargs) -> Iterable[Tuple[List, List, List]]:
    """
    Create a generator that iterates batches for a dataset.

    Overrides the existing ``default_generator`` method to customize how model inputs are
    generated from the data.

    Here, the ``_MapperDMPNN`` helper class is used, for each molecule in a batch, to get required input parameters:

    - atom_features
    - f_ini_atoms_bonds
    - atom_to_incoming_bonds
    - mapping
    - global_features

    Then data from each molecule is converted to a ``_ModData`` object and stored as list of graphs.
    The graphs are modified such that all tensors have same size in 0th dimension. (important requirement for batching)

    Parameters
    ----------
    dataset: Dataset
      the data to iterate
    epochs: int
      the number of times to iterate over the full dataset
    mode: str
      allowed values are 'fit' (called during training), 'predict' (called
      during prediction), and 'uncertainty' (called during uncertainty
      prediction)
    deterministic: bool
      whether to iterate over the dataset in order, or randomly shuffle the
      data for each epoch
    pad_batches: bool
      whether to pad each batch up to this model's preferred batch size

    Returns
    -------
    a generator that iterates batches, each represented as a tuple of lists:
    ([inputs], [outputs], [weights])
    Here, [inputs] is list of graphs.
    """
    for epoch in range(epochs):
      for (X_b, y_b, w_b,
           ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                         deterministic=deterministic,
                                         pad_batches=pad_batches):
        pyg_graphs_list: List = []

        # maximum number of incoming bonds in the batch
        max_num_bonds: int = 1

        for graph in X_b:
          # generate concatenated feature vector and mappings
          mapper: _MapperDMPNN = _MapperDMPNN(graph)
          pyg_graph: _ModData = self._to_pyg_graph(mapper.values)
          max_num_bonds = max(max_num_bonds,
                              pyg_graph['atom_to_incoming_bonds'].shape[1])
          pyg_graphs_list.append(pyg_graph)

        # pad all mappings to maximum number of incoming bonds in the batch
        for graph in pyg_graphs_list:
          required_padding: int = max_num_bonds - graph[
              'atom_to_incoming_bonds'].shape[1]
          graph['atom_to_incoming_bonds'] = nn.functional.pad(
              graph['atom_to_incoming_bonds'], (0, required_padding, 0, 0),
              mode='constant',
              value=-1)
          graph['mapping'] = nn.functional.pad(graph['mapping'],
                                               (0, required_padding, 0, 0),
                                               mode='constant',
                                               value=-1)

        yield (pyg_graphs_list, [y_b], [w_b])
