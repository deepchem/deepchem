import numpy as np
from typing import List, Any
from numpy.typing import ArrayLike
from deepchem.feat.graph_data import BatchGraphData

try:
    import torch
except ModuleNotFoundError:
    pass


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
