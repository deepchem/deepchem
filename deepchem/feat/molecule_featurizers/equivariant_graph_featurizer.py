"""
Featurizer for SE(3)-equivariant Graph Neural Networks.
"""

import numpy as np
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
                 embeded: bool = False):
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

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Featurizes molecules into GraphData objects.

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
            src, dst, edge_features, edge_weights = self._get_fully_connected_edges(
                positions)
        else:
            src, dst, edge_features, edge_weights = self._get_bonded_edges(
                datapoint, positions)

        edge_weights = self._discretize_and_one_hot(edge_weights)
        node_features = np.array(node_features, dtype=np.float32)
        edge_features = np.array(edge_features, dtype=np.float32)

        return GraphData(node_features=node_features,
                         edge_index=np.array([src, dst]),
                         edge_features=edge_features,
                         edge_weights=edge_weights,
                         positions=positions)

    def _get_node_features(self, mol: RDKitMol) -> Tuple[ArrayLike, ArrayLike]:
        """Generates node features and positions.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit Mol object.

        Returns
        -------
        tuple
            A tuple of node features (np.ndarray) and positions (np.ndarray).
        """
        atom_features, positions = [], []

        for atom in mol.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            one_hot = self._one_hot(atomic_number,
                                    [1, 6, 7, 8, 9])  # H, C, N, O, F
            additional_features = [atomic_number]
            atom_features.append(one_hot + additional_features)
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])

        return atom_features, positions

    def _get_bonded_edges(
        self, mol: RDKitMol, positions: np.ndarray
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Generates edges based on bonds.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit Mol object.
        positions : np.ndarray
            Atomic positions.

        Returns
        -------
        tuple
            A tuple of source indices, destination indices, edge features, and edge weights.
        """
        src, dst, edge_features, edge_weights = [], [], [], []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [i, j]
            dst += [j, i]

            dist = positions[j] - positions[i]
            edge_features.append(dist)
            edge_features.append(-dist)
            edge_weights.append([np.linalg.norm(dist)])
            edge_weights.append([np.linalg.norm(dist)])

        return np.array(src), np.array(dst), np.array(edge_features), np.array(
            edge_weights)

    def _get_fully_connected_edges(
        self, positions: np.ndarray
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Generates fully connected edges.

        Parameters
        ----------
        positions : np.ndarray
            Atomic positions.

        Returns
        -------
        tuple
            A tuple of source indices, destination indices, edge features, and edge weights.
        """
        num_atoms = len(positions)
        src, dst, edge_features, edge_weights = [], [], [], []

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
                    dist = positions[j] - positions[i]
                    edge_features.append(dist)
                    edge_weights.append([np.linalg.norm(dist)])

        return np.array(src), np.array(dst), np.array(edge_features), np.array(
            edge_weights)

    def _discretize_and_one_hot(self, edge_weights: ArrayLike) -> np.ndarray:
        """Discretizes edge weights into bins and converts to one-hot encoding.

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
        """Generates a one-hot encoding for a value.

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
