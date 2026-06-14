"""
Tests for SmilesTokenizerV2.

Run with:
    pytest deepchem/feat/tests/test_smiles_tokenizer_v2.py -v
"""

import pytest

from deepchem.feat.smiles_tokenizer_v2 import (
    SmilesTokenizerV2,
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def char_tokenizer():
    return SmilesTokenizerV2(level="char")


@pytest.fixture()
def atom_tokenizer():
    return SmilesTokenizerV2(level="atom")


@pytest.fixture()
def bpe_tokenizer():
    pytest.importorskip("tokenizers")
    smiles_corpus = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "C[NH3+]",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "O=C(O)c1ccccc1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    ]
    t = SmilesTokenizerV2(level="bpe", vocab_size=100)
    t.train(smiles_corpus)
    return t


# ---------------------------------------------------------------------------
# __init__ / invalid level
# ---------------------------------------------------------------------------

class TestInit:
    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level must be"):
            SmilesTokenizerV2(level="word")

    def test_default_level_is_atom(self):
        t = SmilesTokenizerV2()
        assert t.level == "atom"

    def test_repr(self, atom_tokenizer):
        r = repr(atom_tokenizer)
        assert "atom" in r
        assert "SmilesTokenizerV2" in r


# ---------------------------------------------------------------------------
# tokenize – char level
# ---------------------------------------------------------------------------

class TestCharTokenize:
    def test_simple(self, char_tokenizer):
        assert char_tokenizer.tokenize("CCO") == ["C", "C", "O"]

    def test_aspirin(self, char_tokenizer):
        tokens = char_tokenizer.tokenize("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Every character becomes its own token
        assert len(tokens) == len("CC(=O)OC1=CC=CC=C1C(=O)O")

    def test_bracket_not_collapsed(self, char_tokenizer):
        # char level does NOT collapse brackets
        tokens = char_tokenizer.tokenize("[NH3+]")
        assert "[" in tokens and "N" in tokens and "H" in tokens


# ---------------------------------------------------------------------------
# tokenize – atom level
# ---------------------------------------------------------------------------

class TestAtomTokenize:
    def test_chlorine(self, atom_tokenizer):
        assert atom_tokenizer.tokenize("CCl") == ["C", "Cl"]

    def test_bromine(self, atom_tokenizer):
        assert atom_tokenizer.tokenize("CBr") == ["C", "Br"]

    def test_bracket_atom(self, atom_tokenizer):
        tokens = atom_tokenizer.tokenize("C[Cl]")
        assert tokens == ["C", "[Cl]"]

    def test_ammonium(self, atom_tokenizer):
        tokens = atom_tokenizer.tokenize("C[NH3+]")
        assert "[NH3+]" in tokens

    def test_double_bond(self, atom_tokenizer):
        tokens = atom_tokenizer.tokenize("C=O")
        assert "=" in tokens

    def test_ring_closure(self, atom_tokenizer):
        tokens = atom_tokenizer.tokenize("C1CCCCC1")
        assert tokens.count("1") == 2

    def test_stereochemistry(self, atom_tokenizer):
        tokens = atom_tokenizer.tokenize("C[C@@H](O)F")
        # @@  or  @  should appear
        stereo = [t for t in tokens if "@" in t]
        assert len(stereo) >= 1

    def test_aspirin_roundtrip(self, atom_tokenizer):
        smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
        tokens = atom_tokenizer.tokenize(smi)
        assert "".join(tokens) == smi


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_special_tokens_present(self, atom_tokenizer):
        for tok in SPECIAL_TOKENS:
            assert tok in atom_tokenizer.vocab

    def test_pad_id_is_zero(self, atom_tokenizer):
        assert atom_tokenizer.pad_id == 0

    def test_unk_id_is_one(self, atom_tokenizer):
        assert atom_tokenizer.unk_id == 1

    def test_build_vocab_from_smiles(self):
        t = SmilesTokenizerV2(level="atom")
        corpus = ["CCO", "CC(=O)O", "c1ccccc1"]
        t.build_vocab_from_smiles(corpus)
        assert "C" in t.vocab
        assert "c" in t.vocab
        assert "=" in t.vocab


# ---------------------------------------------------------------------------
# encode / decode – char
# ---------------------------------------------------------------------------

class TestCharEncodeDecode:
    def test_encode_returns_list_of_ints(self, char_tokenizer):
        ids = char_tokenizer.encode("CCO")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self, char_tokenizer):
        smi = "CCO"
        ids = char_tokenizer.encode(smi)
        assert char_tokenizer.decode(ids) == smi


# ---------------------------------------------------------------------------
# encode / decode – atom
# ---------------------------------------------------------------------------

class TestAtomEncodeDecode:
    def test_encode_decode_simple(self, atom_tokenizer):
        smi = "CCO"
        assert atom_tokenizer.decode(atom_tokenizer.encode(smi)) == smi

    def test_encode_decode_aspirin(self, atom_tokenizer):
        smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
        assert atom_tokenizer.decode(atom_tokenizer.encode(smi)) == smi

    def test_encode_decode_bracket_atom(self, atom_tokenizer):
        smi = "C[NH3+]"
        assert atom_tokenizer.decode(atom_tokenizer.encode(smi)) == smi

    def test_unk_token_for_unknown(self, atom_tokenizer):
        # 'Ξ' (Greek Xi) is unlikely to be in the default vocabulary
        ids = atom_tokenizer.encode("Ξ")
        assert ids[0] == atom_tokenizer.unk_id

    def test_sos_eos_are_stripped_on_decode(self):
        t = SmilesTokenizerV2(level="atom", add_sos_eos=True)
        smi = "CCO"
        ids = t.encode(smi)
        assert t.decode(ids) == smi  # special tokens stripped

    def test_sos_eos_present_in_id_list(self):
        t = SmilesTokenizerV2(level="atom", add_sos_eos=True)
        ids = t.encode("CCO")
        assert ids[0] == t.sos_id
        assert ids[-1] == t.eos_id


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

class TestPadding:
    def test_padding_to_length(self, atom_tokenizer):
        ids = atom_tokenizer.encode("CCO", pad_length=10)
        assert len(ids) == 10
        # Trailing entries should be pad_id
        assert ids[3:] == [atom_tokenizer.pad_id] * 7

    def test_truncation_to_length(self, atom_tokenizer):
        ids = atom_tokenizer.encode("CC(=O)OC1=CC=CC=C1C(=O)O", pad_length=5)
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Tensor output
# ---------------------------------------------------------------------------

class TestTensorOutput:
    def test_returns_torch_tensor(self, atom_tokenizer):
        pytest.importorskip("torch")
        import torch
        result = atom_tokenizer.encode("CCO", return_tensor=True)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long

    def test_tensor_decode_roundtrip(self, atom_tokenizer):
        pytest.importorskip("torch")
        smi = "CCO"
        tensor = atom_tokenizer.encode(smi, return_tensor=True)
        assert atom_tokenizer.decode(tensor) == smi


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatchEncode:
    def test_batch_padded_equal_lengths(self, atom_tokenizer):
        smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        batch = atom_tokenizer.encode_batch(smiles, pad=True)
        lengths = [len(row) for row in batch]
        assert len(set(lengths)) == 1  # all equal

    def test_batch_no_pad(self, atom_tokenizer):
        smiles = ["CCO", "CC(=O)O"]
        batch = atom_tokenizer.encode_batch(smiles, pad=False)
        assert len(batch[0]) != len(batch[1])  # different lengths

    def test_batch_tensor(self, atom_tokenizer):
        pytest.importorskip("torch")
        import torch
        smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        tensor = atom_tokenizer.encode_batch(smiles, pad=True, return_tensor=True)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 2


# ---------------------------------------------------------------------------
# BPE strategy
# ---------------------------------------------------------------------------

class TestBPE:
    def test_bpe_requires_train(self):
        t = SmilesTokenizerV2(level="bpe")
        with pytest.raises(RuntimeError, match="train"):
            t.tokenize("CCO")

    def test_bpe_train_and_encode(self, bpe_tokenizer):
        ids = bpe_tokenizer.encode("CCO")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_bpe_decode_roundtrip(self, bpe_tokenizer):
        smi = "CCO"
        ids = bpe_tokenizer.encode(smi)
        decoded = bpe_tokenizer.decode(ids)
        assert decoded == smi

    def test_bpe_train_only_for_bpe_level(self):
        t = SmilesTokenizerV2(level="atom")
        with pytest.raises(ValueError, match="level='bpe'"):
            t.train(["CCO"])

    def test_bpe_vocab_populated(self, bpe_tokenizer):
        assert len(bpe_tokenizer.vocab) > len(SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Persistence (save/load vocab)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_vocab(self, atom_tokenizer, tmp_path):
        path = str(tmp_path / "vocab.txt")
        atom_tokenizer.save_vocab(path)
        loaded = SmilesTokenizerV2.load_vocab(path, level="atom")
        assert loaded.vocab == atom_tokenizer.vocab

    def test_loaded_tokenizer_encodes(self, atom_tokenizer, tmp_path):
        path = str(tmp_path / "vocab.txt")
        atom_tokenizer.save_vocab(path)
        loaded = SmilesTokenizerV2.load_vocab(path, level="atom")
        smi = "CC(=O)O"
        assert loaded.decode(loaded.encode(smi)) == smi


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_smiles_char(self, char_tokenizer):
        assert char_tokenizer.tokenize("") == []

    def test_empty_smiles_atom(self, atom_tokenizer):
        assert atom_tokenizer.tokenize("") == []

    def test_single_atom(self, atom_tokenizer):
        assert atom_tokenizer.tokenize("C") == ["C"]

    def test_tensor_input_to_decode(self, atom_tokenizer):
        pytest.importorskip("torch")
        import torch
        ids = atom_tokenizer.encode("CCO")
        tensor = torch.tensor(ids)
        assert atom_tokenizer.decode(tensor) == "CCO"