import os
import json
import numpy as np
from typing import Tuple

from deepchem.utils.data_utils import download_url, get_data_dir
from deepchem.utils.typing import PymatgenStructure
from deepchem.feat import MaterialStructureFeaturizer
from deepchem.feat.graph_data import GraphData

ATOM_INIT_JSON_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/atom_init.json'


class CGCNNFeaturizer(MaterialStructureFeaturizer):
  """
  Calculate structure graph features for crystals.

  Based on the implementation in Crystal Graph Convolutional
  Neural Networks (CGCNN). The method constructs a crystal graph
  representation including atom features and bond features (neighbor
  distances). Neighbors are determined by searching in a sphere around
  atoms in the unit cell. A Gaussian filter is applied to neighbor distances.
  All units are in angstrom.

  This featurizer requires the optional dependency pymatgen. It may
  be useful when 3D coordinates are available and when using graph
  network models and crystal graph convolutional networks.

  See [1]_ for more details.

  References
  ----------
  .. [1] T. Xie and J. C. Grossman, "Crystal graph convolutional
     neural networks for an accurate and interpretable prediction
     of material properties", Phys. Rev. Lett. 120, 2018,
     https://arxiv.org/abs/1710.10324

  Examples
  --------
  >>> import deepchem as dc
  >>> import pymatgen as mg
  >>> featurizer = dc.feat.CGCNNFeaturizer()
  >>> lattice = mg.core.Lattice.cubic(4.2)
  >>> structure = mg.core.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
  >>> features = featurizer.featurize([structure])
  >>> feature = features[0]
  >>> print(type(feature))
  <class 'deepchem.feat.graph_data.GraphData'>

  Note
  ----
  This class requires Pymatgen to be installed.
  """

  def __init__(self,
               radius: float = 8.0,
               max_neighbors: float = 12,
               step: float = 0.2):
    """
    Parameters
    ----------
    radius: float (default 8.0)
      Radius of sphere for finding neighbors of atoms in unit cell.
    max_neighbors: int (default 12)
      Maximum number of neighbors to consider when constructing graph.
    step: float (default 0.2)
      Step size for Gaussian filter. This value is used when building edge features.
    """

    self.radius = radius
    self.max_neighbors = int(max_neighbors)
    self.step = step

    # load atom_init.json
    data_dir = get_data_dir()
    download_url(ATOM_INIT_JSON_URL, data_dir)
    atom_init_json_path = os.path.join(data_dir, 'atom_init.json')
    with open(atom_init_json_path, 'r') as f:
      atom_init_json = json.load(f)

    self.atom_features = {
        int(key): np.array(value, dtype=np.float32)
        for key, value in atom_init_json.items()
    }
    self.valid_atom_number = set(self.atom_features.keys())

  def _featurize(self, datapoint: PymatgenStructure, **kwargs) -> GraphData:
    """
    Calculate crystal graph features from pymatgen structure.

    Parameters
    ----------
    datapoint: pymatgen.core.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    graph: GraphData
      A crystal graph with CGCNN style features.
    """
    if 'struct' in kwargs and datapoint is None:
      datapoint = kwargs.get("struct")
      raise DeprecationWarning(
          'Struct is being phased out as a parameter, please pass "datapoint" instead.'
      )

    node_features = self._get_node_features(datapoint)
    edge_index, edge_features = self._get_edge_features_and_index(datapoint)
    graph = GraphData(node_features, edge_index, edge_features)
    return graph

  def _get_node_features(self, struct: PymatgenStructure) -> np.ndarray:
    """
    Get the node feature from `atom_init.json`. The `atom_init.json` was collected
    from `data/sample-regression/atom_init.json` in the CGCNN repository.

    Parameters
    ----------
    struct: pymatgen.core.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    node_features: np.ndarray
      A numpy array of shape `(num_nodes, 92)`.
    """
    node_features = []
    for site in struct:
      # check whether the atom feature exists or not
      assert site.specie.number in self.valid_atom_number
      node_features.append(self.atom_features[site.specie.number])
    return np.vstack(node_features).astype(float)

  def _get_edge_features_and_index(
      self, struct: PymatgenStructure) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the edge feature and edge index from pymatgen structure.

    Parameters
    ----------
    struct: pymatgen.core.Structure
      A periodic crystal composed of a lattice and a sequence of atomic
      sites with 3D coordinates and elements.

    Returns
    -------
    edge_idx np.ndarray, dtype int
      A numpy array of shape with `(2, num_edges)`.
    edge_features: np.ndarray
      A numpy array of shape with `(num_edges, filter_length)`. The `filter_length` is
      (self.radius / self.step) + 1. The edge features were built by applying gaussian
      filter to the distance between nodes.
    """

    neighbors = struct.get_all_neighbors(self.radius, include_index=True)
    neighbors = [sorted(n, key=lambda x: x[1]) for n in neighbors]

    # construct bi-directed graph
    src_idx, dest_idx = [], []
    edge_distances = []
    for node_idx, neighbor in enumerate(neighbors):
      neighbor = neighbor[:self.max_neighbors]
      src_idx.extend([node_idx] * len(neighbor))
      dest_idx.extend([site[2] for site in neighbor])
      edge_distances.extend([site[1] for site in neighbor])

    edge_idx = np.array([src_idx, dest_idx], dtype=int)
    edge_features = self._gaussian_filter(np.array(edge_distances, dtype=float))
    return edge_idx, edge_features

  def _gaussian_filter(self, distances: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian filter to an array of interatomic distances.

    Parameters
    ----------
    distances : np.ndarray
      A numpy array of the shape `(num_edges, )`.

    Returns
    -------
    expanded_distances: np.ndarray
      Expanded distance tensor after Gaussian filtering.
      The shape is `(num_edges, filter_length)`. The `filter_length` is
      (self.radius / self.step) + 1.
    """

    filt = np.arange(0, self.radius + self.step, self.step)

    # Increase dimension of distance tensor and apply filter
    expanded_distances = np.exp(
        -(distances[..., np.newaxis] - filt)**2 / self.step**2)

    return expanded_distances
