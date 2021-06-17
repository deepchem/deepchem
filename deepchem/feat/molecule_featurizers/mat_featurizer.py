from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
from deepchem.utils.molecule_feature_utils import one_hot_encode
import numpy as np
from rdkit import Chem
from sklearn.metrics import pairwise_distances


class MATFeaturizer(MolecularFeaturizer):
  """
  This class is a featurizer for the Molecule Attention Transformer [1]_.
  The featurizer accepts an RDKit Molecule, and 2 booleans (add_dummy_node and one_hot_formal_charge) as arguments.
  The returned value is a numpy array which consists of molecular graph descriptions:
    - Node Features
    - Adjacency Matrix
    - Distance Matrix

  References
  ---------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer`<https://arxiv.org/abs/2002.08264>`"

  Examples
  --------
  >>> import deepchem as dc
  >>> feat = dc.feat.MATFeaturizer()
  >>> out = feat.featurize("CCC")
  """

  def __init__(
      self,
      add_dummy_node: bool = True,
      one_hot_formal_charge: bool = True,
  ):
    """
    Parameters
    ----------
    add_dummy_node: bool, default True
      If True, a dummy node will be added to the molecular graph.
    one_hot_formal_charge: bool, default True
      If True, formal charges on atoms are one-hot encoded.
    """

    self.add_dummy_node = add_dummy_node
    self.one_hot_formal_charge = one_hot_formal_charge

  def atom_features(self, atom):
    """
    Deepchem already contains an atom_features function, however we are defining a new one here due to the need to handle features specific to MAT.
    Since we need new features like Atom GetNeighbors and IsInRing, and the number of features required for MAT is a fraction of what the Deepchem atom_features function computes, we can speed up computation by defining a custom function.
    
    Parameters
    ----------
    atom: RDKitAtom
      RDKit Atom object.

    Returns
    ----------
    Atom_features: ndarray
      Numpy array containing atom features.
    
    """
    attrib = []
    attrib += one_hot_encode(atom.GetAtomicNum(),
                             [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    attrib += one_hot_encode(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    attrib += one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if self.one_hot_formal_charge:
      attrib += one_hot_encode(atom.GetFormalCharge(), [-1, 0, 1])
    else:
      attrib.append(atom.GetFormalCharge())

    attrib.append(atom.IsInRing())
    attrib.append(atom.GetIsAromatic())

    return np.array(attrib, dtype=np.float32)

  def _featurize(self, mol):
    """
    Featurize the molecule.

    Parameters
    ----------
    mol: RDKitMol
      RDKit mol object.
    
    Returns
    -------
    numpy.ndarray: (node_features, adjacency_matrix, distance_matrix)
    """

    node_features = np.array(
        [self.atom_features(atom) for atom in mol.GetAtoms()])

    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)

    adjacency_matrix.resize(node_features.shape)
    distance_matrix.resize(node_features.shape)

    return node_features, adjacency_matrix, distance_matrix
    