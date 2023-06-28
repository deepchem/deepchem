from rdkit import Chem
import numpy as np
import logging
from typing import List, Optional
from deepchem.utils.typing import RDKitMol

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData

logger = logging.getLogger(__name__)

ATOM_TYPES: dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


def atom_features(datapoint: RDKitMol) -> np.ndarray:
    for atom in datapoint.GetAtoms():
        if atom.GetSymbol() not in ATOM_TYPES.keys():
            raise Exception(
                "We only support 'H', 'C', 'N', 'O' and 'F' at this point for MXMNet Model"
            )

    return np.asarray(
        [[ATOM_TYPES[atom.GetSymbol()]] for atom in datapoint.GetAtoms()],
        dtype=float)


class MXMNetFeaturizer(MolecularFeaturizer):
    """This class is a featurizer for Multiplex Molecular Graph Neural Network (MXMNet) implementation.

    The atomic numbers(indices) of atoms will be used later to generate randomly initialized trainable embeddings to be the input node embeddings.

    This featurizer is based on
    `Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures <https://arxiv.org/pdf/2011.07457.pdf>`_.

    Examples
    --------
    >>> smiles = ["C1=CC=CN=C1", "C1CCC1"]
    >>> featurizer = MXMNetFeaturizer()
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> out[0].num_nodes
    6
    >>> out[0].num_node_features
    1
    >>> out[0].node_features.shape
    (6, 1)
    >>> out[0].num_edges
    12

    Note
    ----
    We are not explitly handling hydrogen atoms for now. We only support 'H', 'C', 'N', 'O' and 'F' atoms to be present in the smiles at this point for MXMNet Model.

    """

    def __init__(self, is_adding_hs: bool = False):
        """
        Parameters
        ----------
        is_adding_hs: bool, default False
            Whether to add Hs or not.

        """
        self.is_adding_hs = is_adding_hs
        super().__init__()

    def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
        """
        Construct edge (bond) index

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        edge_index: np.ndarray
            Edge (Bond) index

        """

        # row, col = edge_index
        src: List[int] = []
        dest: List[int] = []
        for bond in datapoint.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        return np.asarray([src, dest], dtype=int)

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph object with features:
            - node_features: np.ndarray
                    Node feature matrix with shape [num_nodes, num_node_features]
            - edge_index: np.ndarray, dtype int
                    Graph connectivity in COO format with shape [2, num_edges]
            - node_pos_features: np.ndarray, optional (default None)
                    Node position matrix with shape [num_nodes, num_dimensions].
        """

        if isinstance(datapoint, Chem.rdchem.Mol):
            if self.is_adding_hs:
                datapoint = Chem.AddHs(datapoint)
        else:
            raise ValueError(
                "Feature field should contain smiles for MXMNet featurizer!")

        pos: List = []
        pos_x: np.ndarray
        pos_y: np.ndarray
        pos_z: np.ndarray

        # load_sdf_files returns pos as strings but user can also specify
        # numpy arrays for atom coordinates
        if 'pos_x' in kwargs and 'pos_y' in kwargs and 'pos_z' in kwargs:
            if isinstance(kwargs['pos_x'], str):
                pos_x = eval(kwargs['pos_x'])
            elif isinstance(kwargs['pos_x'], np.ndarray):
                pos_x = kwargs['pos_x']
            if isinstance(kwargs['pos_y'], str):
                pos_y = eval(kwargs['pos_y'])
            elif isinstance(kwargs['pos_y'], np.ndarray):
                pos_y = kwargs['pos_y']
            if isinstance(kwargs['pos_z'], str):
                pos_z = eval(kwargs['pos_z'])
            elif isinstance(kwargs['pos_z'], np.ndarray):
                pos_z = kwargs['pos_z']

            for x, y, z in zip(pos_x, pos_y, pos_z):
                pos.append([x, y, z])
            node_pos_features: Optional[np.ndarray] = np.asarray(pos)

        else:
            node_pos_features = None

        # get atom features
        f_atoms: np.ndarray = atom_features(datapoint)

        # get edge index
        edge_index: np.ndarray = self._construct_bond_index(datapoint)

        return GraphData(node_features=f_atoms,
                         edge_index=edge_index,
                         node_pos_features=node_pos_features)
