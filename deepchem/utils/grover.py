import numpy as np
from typing import List, Any
from deepchem.feat.graph_data import BatchGraphData

try:
    import torch
except ModuleNotFoundError:
    pass


def extract_grover_attributes(molgraphs: BatchGraphData):
    fg_labels = getattr(molgraphs, 'fg_labels')
    additional_features = getattr(molgraphs, 'additional_features')
    f_atoms = molgraphs.node_features
    f_bonds = molgraphs.edge_features
    graph_index = molgraphs.graph_index

    # computing atom scopes
    unique_atoms = np.unique(molgraphs.graph_index)
    scopes = []
    for atom in unique_atoms:
        positions = np.where(graph_index == atom, 1, 0)
        scopes.append([np.argmax(positions), np.count_nonzero(positions)])
    a_scope = scopes

    # computing bond scopes
    edge_index = molgraphs.edge_index
    bond_index = graph_index[edge_index[0]]
    scopes = []
    for atom in unique_atoms:
        positions = np.where(bond_index == atom, 1, 0)
        scopes.append([np.argmax(positions), np.count_nonzero(positions)])
    b_scope = scopes

    # computing b2revb
    def find_position(bond, edge_index):
        for i, (sa, da) in enumerate(edge_index.T):
            if sa == bond[1] and da == bond[0]:
                return i

    def compute_b2revb(edge_index):
        b2revb = [0] * len(edge_index[0])
        for i, bond in enumerate(edge_index.T):
            b2revb[i] = find_position(bond, edge_index)
        return b2revb

    b2revb = compute_b2revb(edge_index)

    # computing a2b
    n_atoms = molgraphs.num_nodes
    a2b: List[List[Any]] = [[] for atom in range(n_atoms)]

    for i, bond in enumerate(edge_index.T):
        dest_atom = bond[1]
        a2b[dest_atom].append(i)

    max_num_bonds = max(map(lambda x: len(x), a2b))
    a2b = np.asarray(
        [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])

    f_atoms = torch.FloatTensor(f_atoms)
    f_bonds = torch.FloatTensor(f_bonds)
    fg_labels = torch.FloatTensor(fg_labels)
    additional_features = torch.FloatTensor(additional_features)
    a2b = torch.LongTensor(a2b)
    b2a = torch.LongTensor(molgraphs.edge_index[0])
    b2revb = torch.LongTensor(b2revb)
    # only needed if using atom messages
    a2a = b2a[a2b]  # type: ignore
    a_scope = torch.LongTensor(np.asarray(a_scope))
    b_scope = torch.LongTensor(np.asarray(b_scope))
    return f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, fg_labels, additional_features
