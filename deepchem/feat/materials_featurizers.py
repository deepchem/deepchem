"""
Featurizers for inorganic crystals.
"""

import numpy as np

from deepchem.feat import Featurizer
from deepchem.utils import pad_array

from matminer.featurizers.composition import ElementProperty

from pymatgen import Composition, Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN


class ChemicalFingerprint(Featurizer):
  """
  Chemical fingerprint of elemental properties from composition.

  Parameters
  ----------
  data_source : str, optional (default "matminer")
      Source for element property data ("matminer", "magpie", "deml")
  """

  def __init__(self, data_source='matminer'):
    self.data_source = data_source

  def _featurize(self, comp):
    """
    Calculate chemical fingerprint from crystal composition.

    Parameters
    ----------
    comp : Reduced formula of crystal.
    """

    # Get pymatgen Composition object
    c = Composition(comp)

    ep = ElementProperty.from_preset(self.data_source)

    try:
      feats = ep.featurize(c)
    except:
      feats = []

    return feats


class SineCoulombMatrix(Featurizer):
  """
  Calculate sine Coulomb matrix for crystals.

  Variant of Coulomb matrix for periodic crystals
  Faber et al. (Inter. J. Quantum Chem.
  115, 16, 2015).

  Parameters
  ----------
  max_atoms : int
      Maximum number of atoms for any crystal in the dataset. Used to
      pad the Coulomb matrix.
  eig : bool (default True)
      Return flattened vector of matrix eigenvalues

  """

  def __init__(self, max_atoms, eig=True):
    self.max_atoms = int(max_atoms)
    self.eig = eig

  def _featurize(self, struct):
    """
    Calculate sine Coulomb matrix from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen structure dictionary
    """

    s = Structure.from_dict(struct)
    features = self.sine_coulomb_matrix(s)
    features = np.asarray(features)

    return features

  def sine_coulomb_matrix(self, s):
    """
    Generate sine Coulomb matrices for each crystal.

    Parameters
    ----------
    s : pymatgen structure
    """

    sites = s.sites
    atomic_numbers = np.array([site.specie.Z for site in sites])
    sine_mat = np.zeros((len(sites), len(sites)))
    coords = np.array([site.frac_coords for site in sites])
    lattice = s.lattice.matrix

    # Conversion factor
    ang_to_bohr = 1.8897543760313331

    for i in range(len(sine_mat)):
      for j in range(len(sine_mat)):
        if i == j:
          sine_mat[i][i] = 0.5 * atomic_numbers[i]**2.4
        elif i < j:
          vec = coords[i] - coords[j]
          coord_vec = np.sin(np.pi * vec)**2
          trig_dist = np.linalg.norm(
              (np.matrix(coord_vec) * lattice).A1) * ang_to_bohr
          sine_mat[i][j] = atomic_numbers[i] * atomic_numbers[j] / \
                          trig_dist
        else:
          sine_mat[i][j] = sine_mat[j][i]

    if self.eig:  # flatten array to eigenvalues
      eigs, _ = np.linalg.eig(sine_mat)
      zeros = np.zeros((self.max_atoms,))
      zeros[:len(eigs)] = eigs
      return zeros
    else:
      sine_mat = pad_array(sine_mat, self.max_atoms)
      return sine_mat


class StructureGraphFeaturizer(Featurizer):
  """
  Calculate structure graph for crystals.

  Parameters
  ----------
  strategy : pymatgen.analysis.local_env.NearNeighbors
      An instance of NearNeighbors that determines how graph is constructed.

  #TODO (@ncfrey) process graph features for models
  """

  def __init__(self, strategy=MinimumDistanceNN()):
    self.strategy = strategy

  def _featurize(self, struct):
    """
    Calculate structure graph from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen structure dictionary.
    """

    # Get pymatgen structure object
    s = Structure.from_dict(struct)

    features = self._get_structure_graph_features(s)

    return features

  def _get_structure_graph_features(self, struct):
    """
    Calculate structure graph features from pymatgen structure.

    Parameters
    ----------
    struct : pymatgen structure
    """

    atom_features = np.array([site.specie.Z for site in struct],
                        dtype='int32')

    sg = StructureGraph.with_local_env_strategy(struct, self.strategy)
    nodes = np.asarray(sg.graph.nodes)
    edges = np.asarray(sg.graph.edges)

    return (atom_features, nodes, edges)
