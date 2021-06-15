from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
import numpy as np
from rdkit import Chem
from sklearn.metrics import pairwise_distances


class MATFeaturizer(MolecularFeaturizer):
  """
  This class is a featurizer for the Molecule Attention Transformer [1]_.
  The featurizer accepts an RDKit Molecule, and 2 booleans (add_dummy_node and one_hot_formal_charge) as arguments.
  The returned value is a tuple which consists of molecular graph descriptions:
    - Node Features
    - Adjacency Matrix
    - Distance Matrix
  """

  def __init__(
      self,
      mol: Chem.rdchem.Mol,
      add_dummy_node: bool = True,
      one_hot_formal_charge: bool = True,
  ):
    """
    Parameters
    ----------
    mol (rdchem.Mol): rdkit.Chem.rdchem.Mol
      RDKit Mol object.
    add_dummy_node: bool, default True
      If True, a dummy node will be added to the molecular graph.
    one_hot_formal_charge: bool, default True
      If True, formal charges on atoms are one-hot encoded.

    References
    ---------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer`<https://arxiv.org/abs/2002.08264>`"
    """

    self.mol = mol
    self.add_dummy_node = add_dummy_node
    self.one_hot_formal_charge = one_hot_formal_charge

  def atom_features(self, atom):
    """
    Deepchem already contains an atom_features function, however we are defining a new one here due to the need to handle features specific to MAT.
    Since we need new features like Mol GetNeighbors and IsInRing, and the number of features required for MAT is a fraction of what the Deepchem atom_features function computes, we can speed up computation by defining a custom function.
    
    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object.

    Returns
    ----------
    Atom_features: ndarray
      Numpy array containing atom features.
    
    """
    attrib = []
    attrib += one_hot_encode(atom.GetAtomicNumber(),
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
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object.
    
    Returns
    -------
    tuple: (node_features, adjacency_matrix, distance_matrix)
    """

    node_features = np.array([
        self.atom_features(atom, self.one_hot_formal_charge)
        for atom in mol.getAtoms()
    ])

    adjacency_matrix = Chem.rdmolops.getAdjacencyMatrix(mol)

    conformer = mol.GetConformer()
    positional_matrix = np.array([[
        conformer.GetAtomPosition(k).x,
        conformer.GetAtomPosition(k).y,
        conformer.GetAtomPosition(k).z
    ] for k in range(mol.GetNumAtoms())])
    distance_matrix = pairwise_distances(positional_matrix)

    if self.add_dummy_node:
      m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
      m[1:, 1:] = node_features
      m[0, 0] = 1.0
      node_features = m

      m = np.zeros((adjacency_matrix.shape[0] + 1,
                    adjacency_matrix.shape[1] + 1))
      m[1:, 1:] = adjacency_matrix
      adjacency_matrix = m

      m = np.full((distance_matrix.shape[0] + 1, distance_matrix.shape[1] + 1),
                  1e6)
      m[1:, 1:] = distance_matrix
      distance_matrix = m

    return node_features, adjacency_matrix, distance_matrix