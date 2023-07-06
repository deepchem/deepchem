import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from deepchem.feat.graph_data import GraphData
from deepchem.feat import MolecularFeaturizer

# similar to SNAP featurizer. both taken from Open Graph Benchmark (OGB) github.com/snap-stanford/ogb
# The difference between this and the SNAP features is the lack of masking tokens, possible_implicit_valence_list, possible_bond_dirs
# and the prescence of possible_bond_stereo_list,  possible_is_conjugated_list, possible_is_in_ring_list,
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],  # type: ignore
    'possible_chirality_list': [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER', 'misc'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

full_atom_feature_dims = list(
    map(
        len,  # type: ignore
        [
            allowable_features['possible_atomic_num_list'],
            allowable_features['possible_chirality_list'],
            allowable_features['possible_degree_list'],
            allowable_features['possible_formal_charge_list'],
            allowable_features['possible_numH_list'],
            allowable_features['possible_number_radical_e_list'],
            allowable_features['possible_hybridization_list'],
            allowable_features['possible_is_aromatic_list'],
            allowable_features['possible_is_in_ring_list']
        ]))

full_bond_feature_dims = list(
    map(
        len,  # type: ignore
        [
            allowable_features['possible_bond_type_list'],
            allowable_features['possible_bond_stereo_list'],
            allowable_features['possible_is_conjugated_list']
        ]))


def safe_index(feature_list, e):
    """
    Return index of element e in list l. If e is not present, return the last index

    Parameters
    ----------
    feature_list : list
        Feature vector
    e : int
        Element index to find in feature vector
    """
    try:
        return feature_list.index(e)
    except ValueError:
        return len(feature_list) - 1


class RDKitConformerFeaturizer(MolecularFeaturizer):
    """
    A featurizer that featurizes an RDKit mol object as a GraphData object with 3D coordinates. The 3D coordinates are represented in the node_pos_features attribute of the GraphData object of shape [num_atoms * num_conformers, 3].

    The ETKDGv2 algorithm is used to generate 3D coordinates for the molecule.
    The RDKit source for this algorithm can be found in RDkit/Code/GraphMol/DistGeomHelpers/Embedder.cpp
    The documentation can be found here:
    https://rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.ETKDGv2

    This featurization requires RDKit.

    Examples
    --------
    >>> from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> import numpy as np
    >>> featurizer = RDKitConformerFeaturizer(num_conformers=2)
    >>> molecule = "CCO"
    >>> features_list = featurizer.featurize([molecule])
    >>> batched_feats = BatchGraphData(np.concatenate(features_list).ravel())
    >>> print(batched_feats.node_pos_features.shape)
    (18, 3)
    """

    def __init__(self, num_conformers: int = 1, rmsd_cutoff: float = 2):
        """
        Initialize the RDKitConformerFeaturizer with the given parameters.

        Parameters
        ----------
        num_conformers : int, optional, default=1
            The number of conformers to generate for each molecule.
        rmsd_cutoff : float, optional, default=2
            The root-mean-square deviation (RMSD) cutoff value. Conformers with an RMSD
            greater than this value will be discarded.
        """
        self.num_conformers = num_conformers
        self.rmsd_cutoff = rmsd_cutoff

    def atom_to_feature_vector(self, atom):
        """
        Converts an RDKit atom object to a feature list of indices.

        Parameters
        ----------
        atom : Chem.rdchem.Atom
            RDKit atom object.

        Returns
        -------
        List[int]
            List of feature indices for the given atom.
        """
        atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'],
                       atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'],
                       str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'],
                       atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'],
                       atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'],
                       atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'],
                       atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'],
                       str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(
                atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(
                atom.IsInRing()),
        ]
        return atom_feature

    def bond_to_feature_vector(self, bond):
        """
        Converts an RDKit bond object to a feature list of indices.

        Parameters
        ----------
        bond : Chem.rdchem.Bond
            RDKit bond object.

        Returns
        -------
        List[int]
            List of feature indices for the given bond.
        """
        bond_feature = [
            safe_index(allowable_features['possible_bond_type_list'],
                       str(bond.GetBondType())),
            allowable_features['possible_bond_stereo_list'].index(
                str(bond.GetStereo())),
            allowable_features['possible_is_conjugated_list'].index(
                bond.GetIsConjugated()),
        ]
        return bond_feature

    def _featurize(self, datapoint):
        """
        Featurizes a molecule into a graph representation with 3D coordinates.

        Parameters
        ----------
        datapoint : RdkitMol
            RDKit molecule object

        Returns
        -------
        graph: List[GraphData]
            list of GraphData objects of the molecule conformers with 3D coordinates.
        """
        # add hydrogen bonds to molecule because they are not in the smiles representation
        mol = Chem.AddHs(datapoint)
        ps = AllChem.ETKDGv2()
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.EmbedMultipleConfs(mol, self.num_conformers)
        AllChem.MMFFOptimizeMolecule(mol)
        rmsd_list = []
        rdMolAlign.AlignMolConformers(mol, RMSlist=rmsd_list)
        # insert 0 RMSD for first conformer
        rmsd_list.insert(0, 0)
        conformers = [
            mol.GetConformer(i)
            for i in range(self.num_conformers)
            if rmsd_list[i] < self.rmsd_cutoff
        ]
        # if conformer list is less than num_conformers, pad by repeating conformers
        conf_idx = 0
        while len(conformers) < self.num_conformers:
            conformers.append(conformers[conf_idx])
            conf_idx += 1

        coordinates = [conf.GetPositions() for conf in conformers]

        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_to_feature_vector(atom))

        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = self.bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        graph_list = []
        for i in range(self.num_conformers):
            graph_list.append(
                GraphData(node_pos_features=np.array(coordinates[i]),
                          node_features=np.array(atom_features_list),
                          edge_features=np.array(edge_features_list),
                          edge_index=np.array(edges_list).T))
        return graph_list
