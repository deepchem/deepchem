"""
Test basic molecular features.
"""
import numpy as np
import unittest

from deepchem.feat import RDKitDescriptors


class TestRDKitDescriptors(unittest.TestCase):
    """
    Test RDKitDescriptors.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit.Chem import Descriptors
        self.all_descriptors = Descriptors.descList
        self.all_desc_count = len(self.all_descriptors)
        self.smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'

    def test_rdkit_descriptors(self):
        """
        Test simple descriptors.
        """
        featurizer = RDKitDescriptors()
        descriptors = featurizer.featurize(self.smiles)
        assert descriptors.shape == (1, len(featurizer.reqd_properties))
        exact_mol_wt_index = list(
            featurizer.reqd_properties).index('ExactMolWt')
        assert np.allclose(descriptors[0][exact_mol_wt_index], 180, atol=0.1)

    def test_rdkit_descriptors_with_use_fragment(self):
        """
        Test with use_fragment
        """
        featurizer = RDKitDescriptors(use_fragment=False)
        descriptors = featurizer(self.smiles)
        assert descriptors.shape == (1, len(featurizer.reqd_properties))
        assert len(featurizer.reqd_properties) < self.all_desc_count
        exact_mol_wt_index = list(
            featurizer.reqd_properties).index('ExactMolWt')
        assert np.allclose(descriptors[0, exact_mol_wt_index], 180, atol=0.1)

    def test_rdkit_descriptors_with_use_bcut2d_false(self):
        """
        Test with use_bcut2d
        """
        featurizer = RDKitDescriptors(use_bcut2d=False)
        descriptors = featurizer(self.smiles)
        assert descriptors.shape == (1, len(featurizer.reqd_properties))
        assert len(featurizer.reqd_properties) < self.all_desc_count

        with self.assertRaises(KeyError):
            featurizer.reqd_properties['BCUT2D_MWHI']

        exact_mol_wt_index = list(
            featurizer.reqd_properties).index('ExactMolWt')
        assert np.allclose(descriptors[0, exact_mol_wt_index], 180, atol=0.1)

    def test_rdkit_descriptors_normalized(self):
        """
        Test with normalization
        """
        featurizer = RDKitDescriptors(is_normalized=True)
        assert featurizer.normalized_desc != {}

        descriptors = featurizer(self.smiles)
        assert descriptors.shape == (1, len(featurizer.reqd_properties))

        # no normalized feature value should be greater than 1.0
        assert len(np.where(descriptors > 1.0)[0]) == 0
        exact_mol_wt_index = sorted(
            featurizer.reqd_properties).index('ExactMolWt')
        assert np.allclose(descriptors[0, exact_mol_wt_index], 0.0098, atol=0.1)

    def test_with_custom_descriptors(self):
        # these are the properties used in grover
        grover_props = [
            'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
            'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
            'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
            'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
            'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
            'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
            'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
            'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
            'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
            'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
            'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
            'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
            'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
            'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',
            'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
            'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
            'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
            'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
            'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
            'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
            'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'
        ]
        smiles = 'CCC(=O)'
        featurizer = RDKitDescriptors(descriptors=grover_props,
                                      labels_only=True)
        features = featurizer.featurize(smiles)[0]
        assert len(features) == len(grover_props)
        assert sum(
            features) == 3  # expected number of functional groups in CCC(=O)
        assert (np.where(features == 1)[0] == (10, 11, 23)).all()

    def test_with_labels_only(self):
        descriptors = ['fr_Al_COO', 'fr_Al_OH', 'fr_allylic_oxid']
        smiles = 'CC(C)=CCCC(C)=CC(=O)'
        featurizer = RDKitDescriptors(descriptors=descriptors, labels_only=True)
        features = featurizer.featurize(smiles)[0]
        assert len(features) == len(descriptors)
        assert (features == [0, 0, 1]).all()
