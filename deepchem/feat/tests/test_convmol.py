import numpy as np
from deepchem.feat import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.molnet import load_bace_classification


def get_molecules():
    tasks, all_dataset, transformers = load_bace_classification(
        featurizer="Raw")
    return all_dataset[0].X


def test_mol_ordering():
    mols = get_molecules()
    featurizer = ConvMolFeaturizer()
    featurized_mols = featurizer.featurize(mols)
    for i in range(len(featurized_mols)):
        atom_features = featurized_mols[i].atom_features
        degree_list = np.expand_dims(featurized_mols[i].degree_list, axis=1)
        atom_features = np.concatenate([degree_list, atom_features], axis=1)
        featurized_mols[i].atom_features = atom_features

    conv_mol = ConvMol.agglomerate_mols(featurized_mols)

    for start, end in conv_mol.deg_slice.tolist():
        members = conv_mol.membership[start:end]
        sorted_members = np.array(sorted(members))
        members = np.array(members)
        assert np.all(sorted_members == members)

    conv_mol_atom_features = conv_mol.get_atom_features()

    adj_number = 0
    for start, end in conv_mol.deg_slice.tolist():
        deg_features = conv_mol_atom_features[start:end]
        adj_number_array = deg_features[:, 0]
        assert np.all(adj_number_array == adj_number)
        adj_number += 1
