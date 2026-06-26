from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from deepchem.feat.base_classes import Featurizer
from deepchem.feat.graph_data import GraphData

from ase import Atoms


class AtomisticRadiusGraphFeaturizer(Featurizer):
    """Featurize ``ase.Atoms`` objects as radius graphs.

    This featurizer constructs a non-periodic directed radius graph from
    Cartesian coordinates. Node features are atomic numbers,
    node position features are Cartesian coordinates, edge features are
    displacement vectors, and edge distances are stored as an additional
    ``GraphData`` attribute. For edge ``src -> dst``, ``edge_features`` stores
    ``positions[dst] - positions[src]``. This class requires ASE and accepts
    only ``ase.Atoms`` objects as input.

    Examples
    --------
    >>> import deepchem as dc
    >>> from ase import Atoms  # doctest: +SKIP
    >>> featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=1.5)  # doctest: +SKIP
    >>> atoms = Atoms(numbers=[1, 8], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # doctest: +SKIP
    >>> features = featurizer.featurize([atoms])  # doctest: +SKIP
    >>> graph = features[0]  # doctest: +SKIP
    >>> type(graph)  # doctest: +SKIP
    <class 'deepchem.feat.graph_data.GraphData'>
    """

    def __init__(self, cutoff: float) -> None:
        """
        Parameters
        ----------
        cutoff: float
            Radius cutoff for constructing directed edges. Edges are created
            when the interatomic distance is strictly smaller than this value.
        """
        if cutoff <= 0:
            raise ValueError("cutoff must be greater than 0.")

        self.cutoff = cutoff

    def _get_node_features(self, atoms: "Atoms") -> NDArray[np.int64]:
        """Construct node features from atomic information.

        This helper is separated out so the featurizer can be extended later
        with richer atomic descriptors beyond atomic numbers while keeping the
        graph construction logic unchanged.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic structure represented as an ASE ``Atoms`` object.

        Returns
        -------
        np.ndarray
            Node feature matrix of shape ``(num_atoms, 1)``.
        """
        atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        return atomic_numbers.reshape(-1, 1)

    def _get_radius_graph(
        self, atoms: "Atoms"
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
        """Construct a deterministic directed radius graph from ASE atoms."""
        from ase.neighborlist import neighbor_list

        src_indices, dst_indices, edge_displacements, edge_distances = neighbor_list(
            "ijDd", atoms, self.cutoff, self_interaction=False)

        src_indices = np.asarray(src_indices, dtype=np.int64)
        dst_indices = np.asarray(dst_indices, dtype=np.int64)
        edge_displacements = np.asarray(edge_displacements,
                                        dtype=np.float32).reshape(-1, 3)
        edge_distances = np.asarray(edge_distances,
                                    dtype=np.float32).reshape(-1, 1)

        order = np.lexsort((dst_indices, src_indices))
        src_indices = src_indices[order]
        dst_indices = dst_indices[order]
        edge_displacements = edge_displacements[order]
        edge_distances = edge_distances[order]

        edge_index = np.asarray([src_indices, dst_indices], dtype=np.int64)
        return edge_index, edge_displacements, edge_distances

    def _featurize(self, datapoint: "Atoms", **kwargs: Any) -> GraphData:
        """
        Parameters
        ----------
        datapoint: ase.Atoms
            Atomic structure represented as an ASE ``Atoms`` object.

        Returns
        -------
        GraphData
            Radius graph representation of the atomic structure.
        """
        try:
            from ase import Atoms
        except ModuleNotFoundError:
            raise ImportError("This class requires ASE to be installed.")

        if not isinstance(datapoint, Atoms):
            raise TypeError("datapoint must be an ase.Atoms object.")

        positions = np.asarray(datapoint.get_positions(), dtype=np.float32)
        node_features = self._get_node_features(datapoint)
        edge_index, edge_features, edge_distances_array = self._get_radius_graph(
            datapoint)

        return GraphData(node_features=node_features,
                         edge_index=edge_index,
                         edge_features=edge_features,
                         node_pos_features=positions,
                         edge_distances=edge_distances_array)
