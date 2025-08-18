"""
Featurizer for SE(3)-equivariant Graph Neural Networks.
"""

import numpy as np
from numpy.typing import NDArray
from deepchem.feat.graph_data import GraphData
from deepchem.feat import MolecularFeaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import List, Tuple, Optional, Any
from deepchem.utils.typing import RDKitMol, ArrayLike

logger = logging.getLogger(__name__)


class EquivariantGraphFeaturizer(MolecularFeaturizer):
    """Featurizer for Equivariant Graph Neural Networks.

    This featurizer constructs graph representations of molecular structures,
    capturing atomic features, pairwise distances, and spatial positions. These
    features are tailored for use in Equivariant models with QM9 dataset
    as described in [1]_.

    Features include:
    - **Node features**: Atomic one-hot encodings and additional descriptors.
    - **Edge features**: Vector displacements between atom pairs.
    - **Edge weights**: Discretized pairwise distances in one-hot encoding.
    - **Atomic coordinates**: 3D positions of atoms.

    Examples
    --------
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True, embeded=True)
    >>> features = featurizer.featurize([mol])
    >>> type(features[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> features[0].node_features.shape  # (N, F)
    (3, 6)

    Notes
    -----
    This class requires RDKit to be installed.

    References
    ----------
    .. [1] Fuchs, F. B., et al. "SE(3)-Transformers: 3D Roto-Translation Equivariant
           Attention Networks." arXiv preprint arXiv:2006.10503 (2020).
    """

    def __init__(self,
                 fully_connected: bool = False,
                 weight_bins: Optional[List[Any]] = None,
                 embeded: bool = False,
                 degree: int = 3):
        """
        Parameters
        ----------
        fully_connected : bool, optional (default False)
            If True, generates fully connected graphs with distance-based edges.
        weight_bins : list of float, optional (default [1.0, 2.0, 3.0, 4.0])
            Bin boundaries for discretizing edge weights.
        embeded : bool, optional (default False)
            Whether to embed 3D coordinates using RDKit's ETKDG method.
        """
        self.fully_connected = fully_connected
        self.embeded = embeded
        self.weight_bins = weight_bins if weight_bins is not None else [
            1.0, 2.0, 3.0, 4.0
        ]
        self.degree = degree

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """
        Featurizes molecules into GraphData objects.

        Parameters
        ----------
        datapoint : rdkit.Chem.rdchem.Mol
            RDKit Mol object.

        Returns
        -------
        np.ndarray of GraphData
            List of GraphData objects for the molecule.
        """

        if self.embeded:
            datapoint = Chem.AddHs(datapoint)
            AllChem.EmbedMolecule(datapoint, AllChem.ETKDG())
            datapoint = Chem.RemoveHs(datapoint)

        node_features, positions = self._get_node_features(datapoint)
        positions = np.array(positions, dtype=np.float32)

        if self.fully_connected:
            src, dst, _, edge_weights = self._get_fully_connected_edges(
                positions)
        else:
            src, dst, _, edge_weights = self._get_bonded_edges(datapoint)

        edge_features = positions[src] - positions[dst]

        node_features = np.array(node_features, dtype=np.float32)
        edge_features = np.array(edge_features, dtype=np.float32)

        return GraphData(node_features=node_features,
                         edge_index=np.array([src, dst]),
                         edge_features=edge_features,
                         edge_weights=edge_weights.reshape(-1, 4),
                         node_pos_features=positions)

    def _get_node_features(self, mol: RDKitMol) -> Tuple[ArrayLike, ArrayLike]:
        """
        Generates node features and positions.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit Mol object.

        Returns
        -------
        tuple
            A tuple of node features (np.ndarray) and positions (np.ndarray).
        """
        atom_features = []
        positions: np.ndarray[Any, Any] = np.empty((0, 2), dtype=np.float64)

        for atom in mol.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            one_hot = self._one_hot(atomic_number,
                                    [1, 6, 7, 8, 9])  # H, C, N, O, F
            additional_features = [atomic_number]
            atom_features.append(one_hot + additional_features)

            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            positions = np.array(coords)

        return atom_features, positions

    def _get_bonded_edges(self, mol: RDKitMol):
        """
        Generates edges, edge features, and edge weights for bonded atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit Mol object representing the molecule.

        Returns
        -------
        tuple
            A tuple containing:
            - source (np.ndarray): Indices of source atoms for each edge.
            - destination (np.ndarray): Indices of destination atoms for each edge.
            - edge_features (np.ndarray): Array of edge features (currently empty).
            - edge_weights (np.ndarray): One-hot encoded bond type features for each edge,
            concatenated to account for both directions of the bond.
        """
        Chem.Kekulize(mol, clearAromaticFlags=True)

        BOND_TYPE_TO_INT = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
        }
        NUM_BOND_TYPES = 4

        edge_features: list[float] = []
        edge_weights_list = []
        src, dst = [], []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            src += [i]
            dst += [j]

            bond_type = bond.GetBondType()
            bond_class = BOND_TYPE_TO_INT.get(bond_type, 0)
            one_hot = np.eye(NUM_BOND_TYPES)[bond_class]
            edge_weights_list.append(one_hot)

        edge_weights: NDArray[np.float64] = np.array(edge_weights_list,
                                                     dtype=np.float64)
        edge_weights = np.append(edge_weights, edge_weights, axis=0)
        source = src + dst
        destination = dst + src
        return np.array(source), np.array(destination), np.array(
            edge_features), np.array(edge_weights)

    def _get_fully_connected_edges(self, positions: np.ndarray):
        """
        Generates fully connected edges with one-hot encoded edge weights based on distance bins.

        Parameters
        ----------
        positions : np.ndarray
            Atomic positions.

        Returns
        -------
        tuple
            A tuple of source indices, destination indices, edge features, and edge weights.
        """
        NUM_BOND_TYPES = 4  # here: 4 distance bins
        num_atoms = len(positions)
        src: List[int] = []
        dst: List[int] = []
        edge_features: list[float] = []

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)

        source: np.ndarray = np.array(src)
        dest: np.ndarray = np.array(dst)

        # Compute distance vectors and norms
        dist_vectors = positions[source] - positions[dest]
        distances = np.linalg.norm(dist_vectors, axis=1)

        # Define distance bins (customize as needed)
        bins = np.linspace(0, np.max(distances), NUM_BOND_TYPES + 1)
        bin_indices = np.digitize(distances, bins) - 1
        bin_indices = np.clip(bin_indices, 0, NUM_BOND_TYPES - 1)

        # One-hot encode the distance bins
        edge_weights = np.eye(NUM_BOND_TYPES)[bin_indices]

        return source, dest, np.array(edge_features), edge_weights

    def _discretize_and_one_hot(self, edge_weights: ArrayLike) -> np.ndarray:
        """
        Discretizes edge weights into bins and converts to one-hot encoding.

        Parameters
        ----------
        edge_weights : np.ndarray
            Continuous edge weights to be discretized.

        Returns
        -------
        np.ndarray
            One-hot encoded edge weights.
        """
        edge_weights = np.array(edge_weights).flatten()
        digitized = np.digitize(edge_weights, self.weight_bins, right=False)
        num_bins = len(self.weight_bins) + 1
        one_hot_weights = np.zeros((len(digitized), num_bins))
        one_hot_weights[np.arange(len(digitized)), digitized] = 1
        return one_hot_weights

    def _one_hot(self, value: int, allowable_set: List) -> List[int]:
        """
        Generates a one-hot encoding for a value.

        Parameters
        ----------
        value : int
            Value to encode.
        allowable_set : List
            List of allowable values.

        Returns
        -------
        np.ndarray
            One-hot encoded vector.
        """
        return [1 if value == x else 0 for x in allowable_set]
