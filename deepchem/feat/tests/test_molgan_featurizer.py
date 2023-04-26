import unittest
from deepchem.feat.molecule_featurizers import MolGanFeaturizer
from deepchem.feat.molecule_featurizers import GraphMatrix


class TestMolganFeaturizer(unittest.TestCase):

    def test_featurizer_smiles(self):
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")

        smiles = [
            'Cc1ccccc1CO', 'CC1CCC(C)C(N)C1', 'CCC(N)=O', 'Fc1cccc(F)c1',
            'CC(C)F', 'C1COC2NCCC2C1', 'C1=NCc2ccccc21'
        ]

        invalid_smiles = ['axa', 'xyz', 'inv']

        featurizer = MolGanFeaturizer()
        valid_data = featurizer.featurize(smiles)
        invalid_data = featurizer.featurize(invalid_smiles)

        # test featurization
        valid_graphs = list(
            filter(lambda x: isinstance(x, GraphMatrix), valid_data))
        invalid_graphs = list(
            filter(lambda x: not isinstance(x, GraphMatrix), invalid_data))
        assert len(valid_graphs) == len(smiles)
        assert len(invalid_graphs) == len(invalid_smiles)

        # test defeaturization
        valid_mols = featurizer.defeaturize(valid_graphs)
        invalid_mols = featurizer.defeaturize(invalid_graphs)
        valid_mols = list(
            filter(lambda x: isinstance(x, Chem.rdchem.Mol), valid_mols))
        invalid_mols = list(
            filter(lambda x: not isinstance(x, Chem.rdchem.Mol), invalid_mols))
        assert len(valid_graphs) == len(valid_mols)
        assert len(invalid_graphs) == len(invalid_mols)

        mols = list(map(Chem.MolFromSmiles, smiles))
        redone_smiles = list(map(Chem.MolToSmiles, mols))
        # sanity check; see if something weird does not happen with rdkit
        assert redone_smiles == smiles

        # check if original smiles match defeaturized smiles
        defe_smiles = list(map(Chem.MolToSmiles, valid_mols))
        assert defe_smiles == smiles

    def test_featurizer_rdkit(self):

        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")

        smiles = [
            'Cc1ccccc1CO', 'CC1CCC(C)C(N)C1', 'CCC(N)=O', 'Fc1cccc(F)c1',
            'CC(C)F', 'C1COC2NCCC2C1', 'C1=NCc2ccccc21'
        ]

        invalid_smiles = ['axa', 'xyz', 'inv']

        valid_molecules = list(map(Chem.MolFromSmiles, smiles))
        invalid_molecules = list(map(Chem.MolFromSmiles, invalid_smiles))

        redone_smiles = list(map(Chem.MolToSmiles, valid_molecules))
        # sanity check; see if something weird does not happen with rdkit
        assert redone_smiles == smiles

        featurizer = MolGanFeaturizer()
        valid_data = featurizer.featurize(valid_molecules)
        invalid_data = featurizer.featurize(invalid_molecules)

        # test featurization
        valid_graphs = list(
            filter(lambda x: isinstance(x, GraphMatrix), valid_data))
        invalid_graphs = list(
            filter(lambda x: not isinstance(x, GraphMatrix), invalid_data))
        assert len(valid_graphs) == len(valid_molecules)
        assert len(invalid_graphs) == len(invalid_molecules)

        # test defeaturization
        valid_mols = featurizer.defeaturize(valid_graphs)
        invalid_mols = featurizer.defeaturize(invalid_graphs)
        valid_mols = list(
            filter(lambda x: isinstance(x, Chem.rdchem.Mol), valid_mols))
        invalid_mols = list(
            filter(lambda x: not isinstance(x, Chem.rdchem.Mol), invalid_mols))
        assert len(valid_mols) == len(valid_graphs)
        assert len(invalid_mols) == len(invalid_graphs)

        # check if original smiles match defeaturized smiles
        defe_smiles = list(map(Chem.MolToSmiles, valid_mols))
        assert defe_smiles == smiles


if __name__ == '__main__':
    unittest.main()
