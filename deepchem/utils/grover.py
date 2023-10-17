import numpy as np
from typing import List, Any
from numpy.typing import ArrayLike
from deepchem.feat.graph_data import BatchGraphData, GraphData

try:
    import torch
except ModuleNotFoundError:
    pass

from deepchem.feat.molecule_featurizers.dmpnn_featurizer import GraphConvConstants


class BatchGroverGraph:

    def __init__(self, mol_graphs: List[GraphData]):
        self.smiles_batch = []
        self.features_batch = []
        self.fg_labels = []
        self.additional_features = []
        self.n_mols = len(mol_graphs)

        self.atom_fdim = GraphConvConstants.ATOM_FDIM + 18
        self.bond_fdim = GraphConvConstants.BOND_FDIM + self.atom_fdim

        self.n_atoms = 0
        self.n_bonds = 0
        self.a_scope = [
        ]  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = [
        ]  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        f_atoms = []  # atom features
        f_bonds = []  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [
        ]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond

        for mol_graph in mol_graphs:
            self.smiles_batch.append(mol_graph.smiles)
            self.features_batch.append(mol_graph.additional_features)
            self.fg_labels.append(mol_graph.fg_labels)
            self.additional_features.append(mol_graph.additional_features)
            f_atoms.extend(mol_graph.node_features)
            f_bonds.extend(mol_graph.edge_features)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(np.asarray(f_atoms))
        self.f_bonds = torch.FloatTensor(np.asarray(f_bonds))
        self.a2b = torch.LongTensor(
            np.asarray([
                a2b[a] + [0] * (self.max_num_bonds - len(a2b[a]))
                for a in range(self.n_atoms)
            ]))
        self.b2a = torch.LongTensor(np.asarray(b2a))
        self.b2revb = torch.LongTensor(np.asarray(b2revb))
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = self.b2a[self.a2b]  # only needed if using atom messages
        self.a_scope = torch.LongTensor(self.a_scope)
        self.b_scope = torch.LongTensor(self.b_scope)

        self.fg_labels = torch.Tensor(np.asarray(self.fg_labels)).float()
        self.additional_features = torch.from_numpy(
            np.stack(self.additional_features)).float()

    def get_components(self):
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a2a, self.a_scope, self.b_scope, self.fg_labels


def _get_atom_scopes(graph_index: ArrayLike) -> List[List[int]]:
    """Atom scope is a list of tuples with a single entry for every
    molecule in the batched graph. The entry indicates the beginning
    node index for a molecule and the number of nodes in the molecule.

    Parameters
    ----------
    graph_index: np.array
        An array containing a mapping between node index and the graph
    in the batched graph.

    Returns
    -------
    scopes: List[List[int]]
        Node index scope for each molecule in the batched graph.

    Example
    -------
    >>> import numpy as np
    >>> graph_index = np.array([0, 0, 1, 1, 1])
    >>> _get_atom_scopes(graph_index)
    [[0, 2], [2, 3]]
    """
    # graph_index indicates which atom belongs to which molecule
    mols = np.unique(graph_index)
    scopes = []
    for mol in mols:
        positions = np.where(graph_index == mol, 1, 0)
        scopes.append(
            [int(np.argmax(positions)),
             int(np.count_nonzero(positions))])
    return scopes


def _get_bond_scopes(edge_index: ArrayLike,
                     graph_index: ArrayLike) -> List[List[int]]:
    """Bond scope is a list of tuples with a single entry for every molecule
    in the batched graph. The entry indicates the beginning bond index for a
    molecule and the number of bonds in the molecule.

    Parameters
    ----------
    edge_index: np.array
        Graph connectivity in COO format with shape [2, num_edges]
    graph_index: np.array
        An array containing a mapping between node index and the graph
    in the batched graph.

    Returns
    -------
    scopes: List[List[int]]
        Bond index scope for each molecule in the batched graph.

    Example
    -------
    >>> edge_index = np.array([[0, 1, 2, 4], [1, 0, 4, 2]]) # a molecule with 4 bonds
    >>> graph_index = np.array([0, 0, 1, 1, 1])
    >>> _get_bond_scopes(edge_index, graph_index)
    [[0, 2], [2, 2]]
    """
    mols = np.unique(graph_index)
    bond_index = graph_index[edge_index[0]]  # type: ignore
    scopes = []
    for mol in mols:
        positions = np.where(bond_index == mol, 1, 0)
        scopes.append(
            [int(np.argmax(positions)),
             int(np.count_nonzero(positions))])
    return scopes


def _compute_b2revb(edge_index: np.ndarray) -> List[int]:
    """Every edge in a grover graph is a directed edge. Hence, a bond
    is represented by two edges of opposite directions. b2revb is a representation
    which stores for every edge, the index of reverse edge of that edge.

    Parameters
    ----------
    edge_index: np.array
        Graph connectivity in COO format with shape [2, num_edges]

    Returns
    -------
    b2revb: List[int]
        A mapping where an element at an index contains the index of the reverse bond.

    Example
    -------
    >>> import numpy as np
    >>> edge_index = np.array([[0, 1, 2, 4], [1, 0, 4, 2]])
    >>> _compute_b2revb(edge_index)
    [1, 0, 3, 2]
    """
    b2revb = [0] * edge_index.shape[1]
    for i, bond in enumerate(edge_index.T):
        for j, (sa, da) in enumerate(edge_index.T):
            if sa == bond[1] and da == bond[0]:
                b2revb[i] = j
    return b2revb


def _get_a2b(n_atoms: int, edge_index: np.ndarray) -> np.ndarray:
    """a2b is a mapping between atoms and their incoming bonds.

    Parameters
    ----------
    n_atoms: int
        Number of atoms
    edge_index: np.array
        Graph connectivity in COO format with shape [2, num_edges]

    Returns
    -------
    a2b: ArrayLike
        A mapping between atoms and their incoming bonds

    Example
    -------
    >>> import numpy as np
    >>> edge_index = np.array([[0, 1], [1, 2]])
    >>> n_atoms = 3
    >>> _get_a2b(n_atoms, edge_index)
    array([[0],
           [0],
           [1]])
    """
    a2b: List[List[Any]] = [[] for atom in range(n_atoms)]

    for i, bond in enumerate(edge_index.T):
        dest_atom = bond[1]
        a2b[dest_atom].append(i)

    # padding
    max_num_bonds = max(map(lambda x: len(x), a2b))
    atom_bond_mapping = np.asarray(
        [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])

    return atom_bond_mapping


def extract_grover_attributes(molgraph: BatchGraphData):
    """Utility to extract grover attributes for grover model

    Parameters
    ----------
    molgraph: BatchGraphData
        A batched graph data representing a collection of molecules.

    Returns
    -------
    graph_attributes: Tuple
        A tuple containing atom features, bond features, atom to bond mapping, bond to atom mapping, bond to reverse bond mapping, atom to atom mapping, atom scope, bond scope, functional group labels and other additional features.

    Example
    -------
    >>> import deepchem as dc
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> smiles = ['CC', 'CCC', 'CC(=O)C']
    >>> featurizer = dc.feat.GroverFeaturizer(features_generator=dc.feat.CircularFingerprint())
    >>> graphs = featurizer.featurize(smiles)
    >>> molgraph = BatchGraphData(graphs)
    >>> attributes = extract_grover_attributes(molgraph)
    """
    fg_labels = getattr(molgraph, 'fg_labels')
    additional_features = getattr(molgraph, 'additional_features')
    f_atoms = molgraph.node_features
    f_bonds = molgraph.edge_features
    graph_index = molgraph.graph_index
    edge_index = molgraph.edge_index

    a_scope = _get_atom_scopes(graph_index)
    b_scope = _get_bond_scopes(edge_index, graph_index)
    b2revb = _compute_b2revb(edge_index)

    # computing a2b
    a2b = _get_a2b(molgraph.num_nodes, edge_index)

    f_atoms_tensor = torch.FloatTensor(f_atoms)
    f_bonds_tensor = torch.FloatTensor(f_bonds)
    fg_labels_tensor = torch.FloatTensor(fg_labels)
    additional_features_tensor = torch.FloatTensor(additional_features)
    a2b_tensor = torch.LongTensor(a2b)
    b2a_tensor = torch.LongTensor(molgraph.edge_index[0])
    b2revb_tensor = torch.LongTensor(b2revb)
    # only needed if using atom messages
    a2a = b2a_tensor[a2b_tensor]  # type: ignore
    a_scope_tensor = torch.LongTensor(np.asarray(a_scope))
    b_scope_tensor = torch.LongTensor(np.asarray(b_scope))
    return f_atoms_tensor, f_bonds_tensor, a2b_tensor, b2a_tensor, b2revb_tensor, a2a, a_scope_tensor, b_scope_tensor, fg_labels_tensor, additional_features_tensor
