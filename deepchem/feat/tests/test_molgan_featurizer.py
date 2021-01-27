import unittest
from deepchem.feat.molecule_featurizers import MolGanFeaturizer
from deepchem.feat.molecule_featurizers import GraphMatrix


class TestMolganFeaturizer(unittest.TestCase):

  def test_featurizer_smiles(self):
    smiles = [
        'C#C[C@@]1(C)[NH2+][CH+]N[C@@H]1C', '[NH-][CH+]Oc1nnon1',
        'Cn1ncc2c1C=CC2', 'O=C[C@@H]1[C@H]2[C@H]3[C@@H]1[C@H]1[C@@H]2[N@@H+]31',
        'C#Cc1[nH]cnc1C#N', 'N#C[C@]1(N)CO[C@H]1CO',
        'C[C@@H]1C(NO)C[C@@H]2O[C@@H]21', 'OC1CC=CC1', 'Cn1c(O)ccc1CO',
        '[NH-]C1OC[C@@]2(C=O)N[C@@H]12', 'incorrect smiles'
    ]

    featurizer = MolGanFeaturizer()
    data = featurizer.featurize(smiles)
    incorrect = list(filter(lambda x: not isinstance(x, GraphMatrix), data))
    assert len(data) == len(smiles)
    assert len(incorrect) == 1


if __name__ == '__main__':
  unittest.main()
