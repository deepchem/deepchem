"""Tests for MolecularWeightFeaturizer."""
import pytest
import numpy as np


# ── Basic functionality ────────────────────────────

def test_ethanol_molecular_weight():
    """Ethanol (CCO) has exact MW ≈ 46.04 Da."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    result = f.featurize(["CCO"])
    assert result.shape == (1, 1)
    assert abs(result[0][0] - 46.04) < 0.01


def test_benzene_molecular_weight():
    """Benzene (c1ccccc1) has exact MW ≈ 78.05 Da."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    result = f.featurize(["c1ccccc1"])
    assert abs(result[0][0] - 78.05) < 0.01


# ── Shape & dtype ─────────────────────────────────

def test_output_shape_single():
    """Single molecule → shape (1, 1)."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    result = f.featurize(["CCO"])
    assert result.ndim == 2
    assert result.shape[1] == 1


def test_output_shape_batch():
    """Batch of 3 → shape (3, 1)."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    result = f.featurize(smiles)
    assert result.shape == (3, 1)


def test_output_dtype():
    """Output dtype should be float32."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    result = f.featurize(["CCO"])
    assert result.dtype == np.float32


# ── Edge cases ────────────────────────────────────

def test_all_weights_positive():
    """All valid molecules should yield positive MW."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    smiles = ["C", "CC", "CCO", "c1ccccc1"]
    result = f.featurize(smiles)
    assert (result > 0).all()


def test_invalid_smiles_handled():
    """Invalid SMILES should not raise — base class handles it."""
    from deepchem.feat import MolecularWeightFeaturizer
    f = MolecularWeightFeaturizer()
    result = f.featurize(["INVALID_XYZ_999"])
    assert result.shape == (1, 1)  # zero-filled by base class


# ── Doctest ───────────────────────────────────────

def test_docstring_example():
    """The example in the class docstring should run correctly."""
    from deepchem.feat import MolecularWeightFeaturizer
    featurizer = MolecularWeightFeaturizer()
    features = featurizer.featurize(["CCO", "c1ccccc1"])
    assert features.shape == (2, 1)