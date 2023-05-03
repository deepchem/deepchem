from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from deepchem.feat.graph_data import GraphData
from deepchem.feat import MolecularFeaturizer

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
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
    map(len, [
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
    map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
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
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
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


class Conformer_(MolecularFeaturizer):

    def _featurize(data):
        graphs = []
        for smiles in data:

            mol = Chem.MolFromSmiles(smiles)
            # add hydrogen bonds to molecule because they are not in the smiles representation
            mol = Chem.AddHs(mol)

            try:
                ps = AllChem.ETKDGv2()
                ps.useRandomCoords = True
                AllChem.EmbedMolecule(mol, ps)
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
                conf = mol.GetConformer()
                coordinates = conf.GetPositions()
            except Exception as e:
                print(e)
                print(smiles)
                continue

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))

            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            # Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(edges_list, dtype=torch.long).T
            edge_features = torch.tensor(edge_features_list, dtype=torch.long)

            graph = GraphData(node_pos_features=np.array(coordinates),
                              node_features=np.array(atom_features_list),
                              edge_features=np.array(edge_features),
                              edge_index=np.array(edge_index))
            graphs.append(graph)
        return graphs