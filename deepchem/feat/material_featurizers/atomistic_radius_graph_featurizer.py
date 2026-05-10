import numpy as np

from deepchem.feat.base_classes import Featurizer
from deepchem.feat.graph_data import GraphData


class AtomisticRadiusGraphFeaturizer(Featurizer):
    """Featurize ``ase.Atoms`` objects as radius graphs.

    This featurizer constructs a non-periodic directed radius graph from
    Cartesian coordinates using pure NumPy. Node features are atomic numbers,
    node position features are Cartesian coordinates, edge features are
    displacement vectors, and edge distances are stored as an additional
    ``GraphData`` attribute. For edge ``src -> dst``, ``edge_features`` stores
    ``positions[dst] - positions[src]``. This class requires ASE and accepts
    only ``ase.Atoms`` objects as input.

    Examples
    --------
    >>> import deepchem as dc
    >>> from ase import Atoms
    >>> featurizer = dc.feat.AtomisticRadiusGraphFeaturizer(cutoff=1.5)
    >>> atoms = Atoms(numbers=[1, 8], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> features = featurizer.featurize([atoms])
    >>> graph = features[0]
    >>> type(graph)
    <class 'deepchem.feat.graph_data.GraphData'>
    """

    def __init__(self, cutoff: float):
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

    def _featurize(self, datapoint, **kwargs) -> GraphData:
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

        atomic_numbers = np.asarray(datapoint.get_atomic_numbers(),
                                    dtype=np.int64)
        positions = np.asarray(datapoint.get_positions(), dtype=np.float32)
        num_atoms = len(atomic_numbers)

        src_indices = []
        dst_indices = []
        edge_displacements = []
        edge_distances = []

        for src in range(num_atoms):
            for dst in range(num_atoms):
                if src == dst:
                    continue

                displacement = positions[dst] - positions[src]
                distance = np.linalg.norm(displacement)
                if distance < self.cutoff:
                    src_indices.append(src)
                    dst_indices.append(dst)
                    edge_displacements.append(displacement)
                    edge_distances.append([distance])

        node_features = atomic_numbers.reshape(-1, 1)
        edge_index = np.asarray([src_indices, dst_indices], dtype=np.int64)
        edge_features = np.asarray(edge_displacements,
                                   dtype=np.float32).reshape(-1, 3)
        edge_distances_array = np.asarray(edge_distances,
                                          dtype=np.float32).reshape(-1, 1)

        return GraphData(node_features=node_features,
                         edge_index=edge_index,
                         edge_features=edge_features,
                         node_pos_features=positions,
                         edge_distances=edge_distances_array,
                         atomic_numbers=atomic_numbers)
