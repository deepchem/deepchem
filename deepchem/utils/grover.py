from typing import List, Sequence
from dataclasses import dataclass
import numpy as np
from deepchem.feat.graph_data import GraphData

try:
    import torch
except ModuleNotFoundError:
    pass


@dataclass
class GroverBatchMolGraph:
    """A dataclass for representing Grover Batch Graph"""
    smiles: List[str]
    f_atoms: torch.FloatTensor
    f_bonds: torch.FloatTensor
    fg_labels: torch.FloatTensor
    additional_features: torch.FloatTensor
    a2b: torch.LongTensor
    b2a: torch.LongTensor
    b2revb: torch.LongTensor
    a2a: torch.LongTensor
    a_scope: torch.LongTensor
    b_scope: torch.LongTensor

    def get_components(self):
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.a2a


def grover_batch_mol_graph(
        grover_mol_graphs: Sequence[GraphData]) -> GroverBatchMolGraph:
    """Utility for batching grover molecular graphs.

    A new utility is required here since grover model require additional features
    which has to be padded.

    Example
    -------
    >>> import torch
    >>> from deepchem.utils.grover import grover_batch_mol_graph
    >>> grover_featurizer = dc.feat.GroverFeaturizer(features_generator=dc.feat.CircularFingerprint())
    >>> smiles = ['CC', 'CCC']
    >>> mol_graphs = grover_featurizer.featurize(smiles)
    >>> mol_graph = mol_graphs[0]
    >>> batched_mol_graphs = grover_batch_mol_graph(mol_graphs)

    Parameters
    ----------
    grover_mol_graphs: Sequence[GraphData]
        Accepts a sequence of GraphData objects featurized by GroverFeaturizer.

    Returns
    -------
    batched_graph: GroverBatchMolGraph
        Batched graph data

    """
    # NOTE: This method is similar to batching of graphs for DMPNN model
    # and in future, they should be combined.
    mol_graph_data = grover_mol_graphs[0]
    atom_features_dim = mol_graph_data.node_features.shape[1]
    bond_features_dim = mol_graph_data.edge_features.shape[1]
    # initial number of atoms, bonds (1 b/c of padding) and their features
    n_atoms, n_bonds = 1, 1
    f_atoms, f_bonds = [[0] * atom_features_dim], [[0] * bond_features_dim]

    # list of tuples indicating (start_atom_index, num_atoms), (start_bond_index, num_bonds) for each molecule in batched molecular graph.
    a_scope, b_scope = [], []
    # mapping from atom index to incombing bond indices
    a2b: List[List[int]] = [[]]
    b2a = [
        0
    ]  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = [0]  # mapping from bond index to the index of the reverse bond
    fg_labels = []  # to store functional group labels
    additional_feats = []
    smiles = []

    for mol_graph in grover_mol_graphs:
        smiles.append(getattr(mol_graph, 'smiles'))
        f_atoms.extend(mol_graph.node_features)
        f_bonds.extend(mol_graph.edge_features)
        fg_labels.append(getattr(mol_graph, 'fg_labels'))
        additional_feats.append(getattr(mol_graph, 'additional_features'))

        for a in range(mol_graph.num_nodes):
            # for each atom in the molecule, append incoming bond indices
            # by displacement with n_bonds
            mg_a2b: List = getattr(mol_graph, 'a2b')
            a2b.append([b + n_bonds for b in mg_a2b[a]])

        for b in range(mol_graph.num_edges):
            mg_b2a = getattr(mol_graph, 'b2a')
            mg_b2revb = getattr(mol_graph, 'b2revb')
            b2a.append(n_atoms + mg_b2a[b])
            b2revb.append(n_bonds + mg_b2revb[b])

        a_scope.append((n_atoms, mol_graph.num_nodes))
        b_scope.append((n_bonds, mol_graph.num_edges))

        n_atoms += mol_graph.num_nodes
        n_bonds += mol_graph.num_edges

    # max with 1 to fix a crash in rare case of all single-heavy-atom mols
    max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
    # padding for a2b such that atoms with less than max_num_bonds have 0 padding
    a2b = np.asarray(
        [a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])

    f_atoms = torch.FloatTensor(np.asarray(f_atoms))
    f_bonds = torch.FloatTensor(np.asarray(f_bonds))
    fg_labels = torch.FloatTensor(np.asarray(fg_labels))
    additional_features = torch.FloatTensor(np.asarray(additional_feats))
    a2b = torch.LongTensor(a2b)
    b2a = torch.LongTensor(np.asarray(b2a))
    b2revb = torch.LongTensor(np.asarray(b2revb))
    # only needed if using atom messages
    a2a = b2a[a2b]  # type: ignore
    a_scope = torch.LongTensor(np.asarray(a_scope))
    b_scope = torch.LongTensor(np.asarray(b_scope))
    return GroverBatchMolGraph(smiles, f_atoms, f_bonds, fg_labels,
                               additional_features, a2b, b2a, b2revb, a2a,
                               a_scope, b_scope)
