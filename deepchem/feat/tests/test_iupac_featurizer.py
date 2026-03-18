import logging
import pytest


def test_iupac_featurizer_basic():
    """Test basic featurization of an IUPAC name."""
    from deepchem.feat.molecule_featurizers.iupac_featurizer import IUPACFeaturizer
    featurizer = IUPACFeaturizer(max_len=50)
    result = featurizer.featurize(["ethanol"])
    # featurize stacks uniform-shape outputs into a 2D array
    assert result.shape == (1, 52)


def test_iupac_featurizer_padding():
    """Test that short names are padded to max_len."""
    from deepchem.feat.molecule_featurizers.iupac_featurizer import IUPACFeaturizer
    featurizer = IUPACFeaturizer(max_len=100)
    result = featurizer.featurize(["ethanol"])
    assert result[0].shape == (102,)  # 100 + sos + eos


def test_iupac_featurizer_decode():
    """Test round-trip encode/decode."""
    from deepchem.feat.molecule_featurizers.iupac_featurizer import IUPACFeaturizer
    featurizer = IUPACFeaturizer(max_len=100)
    name = "2-hydroxypropanoic acid"
    encoded = featurizer._featurize(name)
    decoded = featurizer.decode(encoded)
    assert decoded == name


def test_iupac_featurizer_truncation(caplog):
    """Test that names longer than max_len are truncated with a warning."""
    from deepchem.feat.molecule_featurizers.iupac_featurizer import IUPACFeaturizer
    featurizer = IUPACFeaturizer(max_len=5)
    with caplog.at_level(logging.WARNING):
        result = featurizer.featurize(["1,3,7-trimethylpurine-2,6-dione"])
    assert result[0].shape == (7,)  # 5 + sos + eos
    assert "Truncating" in caplog.text


def test_iupac_featurizer_vocab_size():
    """Test vocab_size property."""
    from deepchem.feat.molecule_featurizers.iupac_featurizer import IUPACFeaturizer
    featurizer = IUPACFeaturizer()
    assert featurizer.vocab_size > 0


def test_iupac_featurizer_invalid_input():
    """Test that non-string input raises ValueError."""
    from deepchem.feat.molecule_featurizers.iupac_featurizer import IUPACFeaturizer
    featurizer = IUPACFeaturizer()
    with pytest.raises(ValueError):
        featurizer._featurize(12345)
