"""
Tests for CATH dataset loader.
"""
import os
import unittest
import tempfile
from unittest import mock
from urllib.error import HTTPError

import deepchem as dc

try:
    from Bio.PDB import PDBParser  # noqa: F401
    has_biopython = True
except ImportError:
    has_biopython = False


class TestCATHLoader(unittest.TestCase):
    """Test CATH dataset loader."""

    pdb_content = ("ATOM      1  N   ALA A   1"
                   "       0.000   0.000   0.000  1.00  0.00           N\n"
                   "ATOM      2  CA  ALA A   1"
                   "       1.458   0.000   0.000  1.00  0.00           C\n"
                   "ATOM      3  C   ALA A   1"
                   "       2.009   1.420   0.000  1.00  0.00           C\n"
                   "ATOM      4  N   GLY A   2"
                   "       1.500   2.000   1.000  1.00  0.00           N\n"
                   "ATOM      5  CA  GLY A   2"
                   "       2.000   3.350   1.000  1.00  0.00           C\n"
                   "ATOM      6  C   GLY A   2"
                   "       3.450   3.450   1.500  1.00  0.00           C\n"
                   "END\n")

    def _write_cached_pdbs(self, data_dir, pdb_ids):
        """Write cached PDB files so loader tests do not need the network."""
        cache_dir = os.path.join(data_dir, 'cath_representative_pdb')
        os.makedirs(cache_dir, exist_ok=True)
        for pdb_id in pdb_ids:
            with open(os.path.join(cache_dir, f'{pdb_id.lower()}.pdb'),
                      'w') as pdb_file:
                pdb_file.write(self.pdb_content)

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_default(self):
        """Test loading CATH dataset from cached PDB files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_ids = ['1AAA', '1AAB', '1AAC']
            self._write_cached_pdbs(tmpdir, pdb_ids)
            with mock.patch(
                    'deepchem.molnet.load_function.cath_datasets.urlopen'
            ) as mock_urlopen:
                tasks, datasets, transformers = dc.molnet.load_cath(
                    featurizer='ProteinBackbone',
                    splitter='random',
                    data_dir=tmpdir,
                    save_dir=tmpdir,
                    reload=False,
                    pdb_ids=pdb_ids)
            mock_urlopen.assert_not_called()

            # Check tasks
            assert len(tasks) == 1
            assert tasks[0] == 'structure_placeholder'

            # Check datasets
            train, valid, test = datasets
            assert isinstance(train, dc.data.Dataset)
            assert isinstance(valid, dc.data.Dataset)
            assert isinstance(test, dc.data.Dataset)

            # Check that we have some data
            total_size = len(train) + len(valid) + len(test)
            assert total_size == len(pdb_ids)

            # Check feature shape
            if len(train) > 0:
                sample = train.X[0]
                assert sample.ndim == 3  # (L, 3, 3)
                assert sample.shape[1] == 3  # N, CA, C
                assert sample.shape[2] == 3  # x, y, z

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_no_split(self):
        """Test loading CATH without splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_ids = ['1AAA', '1AAB']
            self._write_cached_pdbs(tmpdir, pdb_ids)
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=False,
                pdb_ids=pdb_ids)

            # Should have single dataset
            assert len(datasets) == 1
            dataset = datasets[0]
            assert isinstance(dataset, dc.data.DiskDataset)
            assert len(dataset) == len(pdb_ids)
            assert list(dataset.ids) == pdb_ids

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_max_length(self):
        """Test loading CATH with custom max_length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_ids = ['1AAA']
            self._write_cached_pdbs(tmpdir, pdb_ids)
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                data_dir=tmpdir,
                save_dir=tmpdir,
                max_length=256,
                reload=False,
                pdb_ids=pdb_ids)

            dataset = datasets[0]
            if len(dataset) > 0:
                # Check that proteins are not longer than max_length
                for i in range(len(dataset)):
                    assert dataset.X[i].shape[0] <= 256

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_reload(self):
        """Test reloading cached CATH dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_ids = ['1AAA', '1AAB']
            self._write_cached_pdbs(tmpdir, pdb_ids)
            # First load
            tasks1, datasets1, _ = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=True,
                pdb_ids=pdb_ids)

            # Second load (should reload from cache)
            tasks2, datasets2, _ = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=True,
                pdb_ids=pdb_ids)

            # Should have same number of samples
            assert len(datasets1[0]) == len(datasets2[0])
            assert tasks1 == tasks2
            assert isinstance(datasets2[0], dc.data.DiskDataset)

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_cath_loader_class(self):
        """Test _CATHLoader class directly."""
        from deepchem.molnet.load_function.cath_datasets import _CATHLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = dc.feat.ProteinBackboneFeaturizer()
            loader = _CATHLoader(featurizer=featurizer,
                                 splitter=None,
                                 transformer_generators=[],
                                 tasks=['structure_placeholder'],
                                 data_dir=tmpdir,
                                 save_dir=tmpdir,
                                 max_length=512,
                                 pdb_ids=['1AAA'])

            assert loader.name == 'cath_representative_pdb'
            assert loader.max_length == 512

            # Test PDB list
            pdb_list = loader._get_cath_pdb_list()
            assert pdb_list == ['1AAA']

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_invalid_loader_inputs(self):
        """Test invalid loader inputs fail clearly."""
        from deepchem.molnet.load_function.cath_datasets import _CATHLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = dc.feat.ProteinBackboneFeaturizer()
            with self.assertRaises(ValueError):
                _CATHLoader(featurizer=featurizer,
                            splitter=None,
                            transformer_generators=[],
                            tasks=['structure_placeholder'],
                            data_dir=tmpdir,
                            save_dir=tmpdir,
                            max_length=0)
            with self.assertRaises(ValueError):
                _CATHLoader(featurizer=featurizer,
                            splitter=None,
                            transformer_generators=[],
                            tasks=['structure_placeholder'],
                            data_dir=tmpdir,
                            save_dir=tmpdir,
                            pdb_ids=[])

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_missing_download_raises(self):
        """Test missing PDB downloads fail explicitly."""
        from deepchem.molnet.load_function.cath_datasets import _CATHLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = dc.feat.ProteinBackboneFeaturizer()
            loader = _CATHLoader(featurizer=featurizer,
                                 splitter=None,
                                 transformer_generators=[],
                                 tasks=['structure_placeholder'],
                                 data_dir=tmpdir,
                                 save_dir=tmpdir,
                                 pdb_ids=['1AAA'])

            with mock.patch(
                    'deepchem.molnet.load_function.cath_datasets.urlopen',
                    side_effect=HTTPError('http://example.com/1aaa.pdb', 404,
                                          'Not Found', None, None)):
                with self.assertRaisesRegex(ValueError,
                                            'Failed to download PDB IDs: 1AAA'):
                    loader._download_pdbs(['1AAA'])


if __name__ == '__main__':
    unittest.main()
