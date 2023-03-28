import numpy as np
from typing import List, Any
from deepchem.feat.graph_data import BatchGraphData

try:
    import torch
except ModuleNotFoundError:
    pass


def _get_atom_scopes(graph_index):
    """Atom scope is a list of tuples with a single entry for every
    molecule in the batched graph. The entry indicates the beginning
    node index for a molecule and the number of nodes in the molecule.

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
        scopes.append([np.argmax(positions), np.count_nonzero(positions)])
    return scopes


def _get_bond_scopes(edge_index, graph_index):
    """Bond scope is a list of tuples with a single entry for every molecule
    in the batched graph. The entry indicates the beginning bond index for a
    molecule and the number of bonds in the molecule.

    Example
    -------
    >>> edge_index = np.array([[0, 1, 2, 4], [1, 0, 4, 2]]) # a molecule with 4 bonds
    >>> graph_index = np.array([0, 0, 1, 1, 1])
    >>> _get_bond_scopes(edge_index, graph_index)
    [[0, 2], [2, 2]]
    """
    mols = np.unique(graph_index)
    bond_index = graph_index[edge_index[0]]
    scopes = []
    for mol in mols:
        positions = np.where(bond_index == mol, 1, 0)
        scopes.append([np.argmax(positions), np.count_nonzero(positions)])
    return scopes


def _compute_b2revb(edge_index):
    """Every edge in a grover graph is a directed edge. Hence, a bond
    is represented by two edges of opposite directions. b2revb is a representation
    which stores for every edge, the index of reverse edge of that edge.

    Example
    -------
    >>> import numpy as np
    >>> edge_index = np.array([[0, 1, 2, 4], [1, 0, 4, 2]])
    >>> _compute_b2revb(edge_index)
    [1, 0, 3, 2]
    """
    b2revb = [0] * len(edge_index[0])
    for i, bond in enumerate(edge_index.T):
        for j, (sa, da) in enumerate(edge_index.T):
            if sa == bond[1] and da == bond[0]:
                b2revb[i] = j
    return b2revb


def _get_a2b(n_atoms, edge_index):
    """a2b is a mapping between atoms and their incoming bonds

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
    a2b = np.asarray(
        [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])

    return a2b


def extract_grover_attributes(molgraph: BatchGraphData):
    """Utility to extract grover attributes for grover model

    Parameter
    ---------
    molgraph: BatchGraphData
        A batched graph data representing a collection of molecules.

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

    f_atoms = torch.FloatTensor(f_atoms)
    f_bonds = torch.FloatTensor(f_bonds)
    fg_labels = torch.FloatTensor(fg_labels)
    additional_features = torch.FloatTensor(additional_features)
    a2b = torch.LongTensor(a2b)
    b2a = torch.LongTensor(molgraph.edge_index[0])
    b2revb = torch.LongTensor(b2revb)
    # only needed if using atom messages
    a2a = b2a[a2b]  # type: ignore
    a_scope = torch.LongTensor(np.asarray(a_scope))
    b_scope = torch.LongTensor(np.asarray(b_scope))
    return f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels, additional_features
