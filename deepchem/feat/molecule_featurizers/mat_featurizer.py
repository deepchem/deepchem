from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.typing import RDKitMol, RDKitAtom
import numpy as np
from typing import Tuple, Any
from dataclasses import dataclass


@dataclass
class MATEncoding:
    """
    Dataclass specific to the Molecular Attention Transformer [1]_.

    This dataclass class wraps around three different matrices for a given molecule: Node Features, Adjacency Matrix, and the Distance Matrix.

    Parameters
    ----------
    node_features: np.ndarray
        Node Features matrix for the molecule. For MAT, derived from the construct_node_features_matrix function.
    adjacency_matrix: np.ndarray
        Adjacency matrix for the molecule. Derived from rdkit.Chem.rdmolops.GetAdjacencyMatrix
    distance_matrix: np.ndarray
        Distance matrix for the molecule. Derived from rdkit.Chem.rdmolops.GetDistanceMatrix

    """
    node_features: np.ndarray
    adjacency_matrix: np.ndarray
    distance_matrix: np.ndarray


class MATFeaturizer(MolecularFeaturizer):
    """
    This class is a featurizer for the Molecule Attention Transformer [1]_.
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

    Note
    ----
    This class requires RDKit to be installed.

    """

    def __init__(self):
        pass

    def construct_mol(self, mol: RDKitMol) -> RDKitMol:
        """
        Processes an input RDKitMol further to be able to extract id-specific Conformers from it using mol.GetConformer().

        Parameters
        ----------
        mol: RDKitMol
          RDKit Mol object.

        Returns
        ----------
        mol: RDKitMol
            A processed RDKitMol object which is embedded, UFF Optimized and has Hydrogen atoms removed. If the former conditions are not met and there is a value error, then 2D Coordinates are computed instead.

        """
        try:
            from rdkit.Chem import AllChem
            from rdkit import Chem
        except ModuleNotFoundError:
            pass
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except ValueError:
            AllChem.Compute2DCoords(mol)

        return mol

    def atom_features(self, atom: RDKitAtom) -> np.ndarray:
        """Deepchem already contains an atom_features function, however we are defining a new one here due to the need to handle features specific to MAT.
        Since we need new features like Atom GetNeighbors and IsInRing, and the number of features required for MAT is a fraction of what the Deepchem atom_features function computes, we can speed up computation by defining a custom function.

        Parameters
        ----------
        atom: RDKitAtom
            RDKit Atom object.

        Returns
        ----------
        ndarray
            Numpy array containing atom features.

        """
        attrib = []
        attrib += one_hot_encode(atom.GetAtomicNum(),
                                 [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
        attrib += one_hot_encode(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
        attrib += one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        attrib += one_hot_encode(atom.GetFormalCharge(),
                                 [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

        attrib.append(atom.IsInRing())
        attrib.append(atom.GetIsAromatic())

        return np.array(attrib, dtype=np.float32)

    def construct_node_features_matrix(self, mol: RDKitMol) -> np.ndarray:
        """This function constructs a matrix of atom features for all atoms in a given molecule using the atom_features function.

        Parameters
        ----------
        mol: RDKitMol
            RDKit Mol object.

        Returns
        ----------
        Atom_features: ndarray
            Numpy array containing atom features.

        """
        return np.array([self.atom_features(atom) for atom in mol.GetAtoms()])

    def _add_dummy_node(
            self, node_features: np.ndarray, adj_matrix: np.ndarray,
            dist_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adds a single dummy node to the molecule, which is consequently reflected in the Node Features Matrix, Adjacency Matrix and the Distance Matrix.

        Parameters
        ----------
        node_features: np.ndarray
            Node Features matrix for a given molecule.
        adjacency_matrix: np.ndarray
            Adjacency matrix for a given molecule.
        distance_matrix: np.ndarray
            Distance matrix for a given molecule.

        Returns
        ----------
        Atom_features: Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing three numpy arrays: node_features, adjacency_matrix, distance_matrix.

        """

        if node_features is not None:
            m = np.zeros(
                (node_features.shape[0] + 1, node_features.shape[1] + 1))
            m[1:, 1:] = node_features
            m[0, 0] = 1.0
            node_features = m

        if adj_matrix is not None:
            m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
            m[1:, 1:] = adj_matrix
            adj_matrix = m

        if dist_matrix is not None:
            m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1),
                        1e6)
            m[1:, 1:] = dist_matrix
            dist_matrix = m

        return node_features, adj_matrix, dist_matrix

    def _pad_array(self, array: np.ndarray, shape: Any) -> np.ndarray:
        """Pads an array to the desired shape.

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

    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pads a given sequence using the pad_array function.

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
        return np.stack([self._pad_array(t, shape=max_shape) for t in sequence])

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> MATEncoding:
        """
        Featurize the molecule.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        MATEncoding
            A MATEncoding dataclass instance consisting of processed node_features, adjacency_matrix and distance_matrix.

        """
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        from rdkit import Chem

        datapoint = self.construct_mol(datapoint)

        node_features = self.construct_node_features_matrix(datapoint)
        adjacency_matrix = Chem.GetAdjacencyMatrix(datapoint)
        distance_matrix = Chem.GetDistanceMatrix(datapoint)

        node_features, adjacency_matrix, distance_matrix = self._add_dummy_node(
            node_features, adjacency_matrix, distance_matrix)

        node_features = self._pad_sequence(node_features)
        adjacency_matrix = self._pad_sequence(adjacency_matrix)
        distance_matrix = self._pad_sequence(distance_matrix)

        return MATEncoding(node_features, adjacency_matrix, distance_matrix)
