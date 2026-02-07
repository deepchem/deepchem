"""
Tests for CATH dataset loader.
"""
import unittest
import tempfile
import pytest
import deepchem as dc

try:
    from Bio.PDB import PDBParser
    has_biopython = True
except ImportError:
    has_biopython = False


class TestCATHLoader(unittest.TestCase):
    """Test CATH dataset loader."""

    @pytest.mark.slow
    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_default(self):
        """Test loading CATH dataset with default parameters."""
        # Use a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter='random',
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=False)

            # Check tasks
            assert len(tasks) == 1
            assert tasks[0] == 'fold_class'

            # Check datasets
            train, valid, test = datasets
            assert isinstance(train, dc.data.Dataset)
            assert isinstance(valid, dc.data.Dataset)
            assert isinstance(test, dc.data.Dataset)

            # Check that we have some data
            total_size = len(train) + len(valid) + len(test)
            assert total_size > 0

            # Check feature shape
            if len(train) > 0:
                sample = train.X[0]
                assert sample.ndim == 3  # (L, 3, 3)
                assert sample.shape[1] == 3  # N, CA, C
                assert sample.shape[2] == 3  # x, y, z

    @pytest.mark.slow
    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_no_split(self):
        """Test loading CATH without splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=False)

            # Should have single dataset
            assert len(datasets) == 1
            dataset = datasets[0]
            assert isinstance(dataset, dc.data.Dataset)
            assert len(dataset) > 0

    @pytest.mark.slow
    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_max_length(self):
        """Test loading CATH with custom max_length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks, datasets, transformers = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter=None,
                data_dir=tmpdir,
                save_dir=tmpdir,
                max_length=256,
                reload=False)

            dataset = datasets[0]
            if len(dataset) > 0:
                # Check that proteins are not longer than max_length
                for i in range(len(dataset)):
                    assert dataset.X[i].shape[0] <= 256

    @pytest.mark.slow
    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_load_cath_reload(self):
        """Test reloading cached CATH dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First load
            tasks1, datasets1, _ = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter='random',
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=True)

            # Second load (should reload from cache)
            tasks2, datasets2, _ = dc.molnet.load_cath(
                featurizer='ProteinBackbone',
                splitter='random',
                data_dir=tmpdir,
                save_dir=tmpdir,
                reload=True)

            # Should have same number of samples
            assert len(datasets1[0]) == len(datasets2[0])

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_cath_loader_class(self):
        """Test _CATHLoader class directly."""
        from deepchem.molnet.load_function.cath_datasets import _CATHLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = dc.feat.ProteinBackboneFeaturizer()
            loader = _CATHLoader(featurizer=featurizer,
                                 splitter=None,
                                 transformers=[],
                                 tasks=['fold_class'],
                                 data_dir=tmpdir,
                                 save_dir=tmpdir,
                                 max_length=512)

            assert loader.name == 'cath_s40'
            assert loader.max_length == 512

            # Test PDB list
            pdb_list = loader._get_cath_pdb_list()
            assert isinstance(pdb_list, list)
            assert len(pdb_list) > 0
            assert all(isinstance(pdb, str) for pdb in pdb_list)


if __name__ == '__main__':
    unittest.main()
