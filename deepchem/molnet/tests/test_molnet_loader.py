"""
Tests for common molnet loader functionality.
"""
import os
import string
import tempfile

import numpy as np

import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import (
    _MolnetLoader, _get_safe_directory_name)


class _ToyLoader(_MolnetLoader):
    """Loader which builds a small dataset in memory instead of downloading."""

    def create_dataset(self) -> dc.data.Dataset:
        smiles = ['C', 'CC', 'CCC', 'CCO']
        X = self.featurizer.featurize(smiles)
        y = np.arange(len(smiles), dtype=np.float64).reshape(-1, 1)
        return dc.data.DiskDataset.from_numpy(X, y, ids=smiles)


def test_safe_directory_name_keeps_short_names():
    """Short names without unsafe characters are left unchanged."""
    name = 'CircularFingerprint_radius_4_size_1024'
    assert _get_safe_directory_name(name) == name


def test_safe_directory_name_shortens_long_names():
    """Long names are truncated and distinct names stay distinct."""
    name1 = 'Featurizer_' + 'a' * 1000
    name2 = 'Featurizer_' + 'a' * 999 + 'b'
    safe1 = _get_safe_directory_name(name1)
    safe2 = _get_safe_directory_name(name2)
    assert len(safe1) == 128
    assert len(safe2) == 128
    assert safe1.startswith('Featurizer_')
    assert safe1 != safe2
    # The result is deterministic, so cached datasets can be reloaded.
    assert safe1 == _get_safe_directory_name(name1)


def test_safe_directory_name_removes_unsafe_characters():
    """Characters which are invalid in paths are replaced."""
    safe = _get_safe_directory_name("Featurizer_a_{'/': 1, ':': 2}")
    for char in '<>:"/\\|?*':
        assert char not in safe


def test_load_dataset_with_long_featurizer_name():
    """A featurizer with a very long str() can be loaded and reloaded.

    See https://github.com/deepchem/deepchem/issues/2813
    """
    char_to_idx = {c: i for i, c in enumerate(string.printable)}
    featurizer = dc.feat.SmilesToSeq(char_to_idx=char_to_idx)
    assert len(str(featurizer)) > 255
    save_dir = tempfile.mkdtemp()
    loader = _ToyLoader(featurizer,
                        None, [], ['task'],
                        data_dir=tempfile.mkdtemp(),
                        save_dir=save_dir)
    tasks, (dataset,), _ = loader.load_dataset('toy', reload=True)
    assert tasks == ['task']
    assert len(dataset) == 4
    featurized_dir = os.path.join(save_dir, 'toy-featurized')
    assert os.listdir(featurized_dir) == [
        _get_safe_directory_name(str(featurizer))
    ]

    # The second call reloads the cached dataset from the same directory.
    _, (reloaded,), _ = loader.load_dataset('toy', reload=True)
    assert reloaded.data_dir == dataset.data_dir
    np.testing.assert_array_equal(reloaded.y, dataset.y)
