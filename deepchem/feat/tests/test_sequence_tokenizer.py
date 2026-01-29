import numpy as np
from deepchem.feat.sequence_tokenizer import DNACharTokenizer, DNAKmerTokenizer


def test_dna_char_tokenizer_basic():
    tokenizer = DNACharTokenizer()
    out = tokenizer.featurize(["ACGT", "A"])

    assert out["input_ids"].shape == (2, 4)
    assert out["attention_mask"].shape == (2, 4)

    # First sequence fully filled
    assert np.all(out["attention_mask"][0] == np.array([1, 1, 1, 1]))
    # Second sequence padded
    assert np.all(out["attention_mask"][1] == np.array([1, 0, 0, 0]))


def test_dna_char_tokenizer_max_length():
    tokenizer = DNACharTokenizer(max_length=2)
    out = tokenizer.featurize(["ACGT"])

    assert out["input_ids"].shape == (1, 2)
    assert np.all(out["attention_mask"] == np.array([[1, 1]]))


def test_dna_kmer_tokenizer_basic():
    tokenizer = DNAKmerTokenizer(k=2)
    out = tokenizer.featurize(["ACGT"])

    # ACGT -> AC, CG, GT (3 tokens)
    assert out["input_ids"].shape == (1, 3)
    assert np.all(out["attention_mask"] == np.array([[1, 1, 1]]))


def test_dna_kmer_padding():
    tokenizer = DNAKmerTokenizer(k=2, max_length=5)
    out = tokenizer.featurize(["ACGT"])

    assert out["input_ids"].shape == (1, 5)
    assert np.all(out["attention_mask"] == np.array([[1, 1, 1, 0, 0]]))
