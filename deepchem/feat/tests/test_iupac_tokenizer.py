"""Tests for IUPACTokenizer."""
import pytest
from deepchem.feat.molecule_featurizers.iupac_tokenizer import IUPACTokenizer


@pytest.fixture
def tok():
    return IUPACTokenizer()


def test_tokenize_simple(tok):
    # Full alkane name matched as a single token; connecting form + suffix split
    assert tok.tokenize("methane") == ["methane"]
    assert tok.tokenize("ethanol") == ["ethan", "ol"]


def test_tokenize_complex(tok):
    assert tok.tokenize("2,4,6-trinitrotoluene") == [
        "2", ",", "4", ",", "6", "-", "tri", "nitro", "toluene"
    ]
    assert tok.tokenize("(2R)-2-methylbutanoic acid") == [
        "(", "2", "R", ")", "-", "2", "-", "methyl", "butan", "oic acid"
    ]


def test_stereo_descriptors(tok):
    for stereo in ["(R)", "(S)", "(E)", "(Z)", "(+)", "(-)"]:
        assert tok.tokenize(stereo) == [stereo], f"{stereo!r} must be a single token"


def test_build_vocab(tok):
    tok.build_vocab(["methane", "ethanol"])
    assert tok.vocab["[UNK]"] == 0
    assert len(tok.vocab) > 1
    # all IDs must be unique
    assert len(set(tok.vocab.values())) == len(tok.vocab)


def test_encode_decode(tok):
    name = "2-methylpropan-1-ol"
    tok.build_vocab([name])
    assert tok.decode(tok.encode(name)) == name


def test_unknown_token(tok):
    tok.build_vocab(["methane"])
    # "XYZ" has no tokens in vocab → all IDs should be [UNK] = 0
    assert all(i == 0 for i in tok.encode("XYZ"))


def test_empty_string(tok):
    assert tok.tokenize("") == []
    assert tok.encode("") == []
