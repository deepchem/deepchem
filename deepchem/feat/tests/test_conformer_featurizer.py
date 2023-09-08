from rdkit import Chem


def test_conformer_featurizer():
    from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    smiles = "C1=CC=NC=C1"
    featurizer = RDKitConformerFeaturizer()
    conformer = featurizer.featurize(smiles)[0]

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    assert conformer.num_nodes == num_atoms
    # bond and its reverse bond, therefore num_bonds * 2 edges
    assert conformer.num_edges == num_bonds * 2
    # 9 atom features
    assert conformer.node_features.shape == (num_atoms, 9)
    # Graph connectivity in COO format with shape [2, num_edges]
    assert conformer.edge_index.shape == (2, num_bonds * 2)
    # 3 bond features
    assert conformer.edge_features.shape == (num_bonds * 2, 3)
    # 3 xyz coordinates for each atom in the conformer
    assert conformer.node_pos_features.shape == (num_atoms, 3)
    assert conformer.num_edge_features == 3
    assert conformer.num_node_features == 9
