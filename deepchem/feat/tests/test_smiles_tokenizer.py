import unittest
import os
import tempfile
import json
from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class TestSmilesTokenizer(unittest.TestCase):
    """Tests the production-grade SmilesTokenizer for DeepChem."""

    def setUp(self):
        self.tokenizer = SmilesTokenizer(level="atom")
        self.smiles_list = [
            "CCO", "C[Cl]", "[NH3+]CC", "C[C@H](O)C", "c1ccccc1", "CC(=O)O", "C1CO1",
            "C12=CC=CC=C1C=CC=C2"
        ]
        self.tokenizer.train(self.smiles_list)

    def test_atom_parsing_accuracy(self):
        """Verify accurate atom-level tokenization for complex chemistry."""
        # Brackets and stereochemistry
        smiles = "C[C@H](O)C"
        tokens = self.tokenizer.encode(smiles, add_special_tokens=False)
        self.assertEqual(tokens, ["C", "[C@H]", "(", "O", ")", "C"])

        # Multi-character atoms
        smiles2 = "C[Cl]"
        tokens2 = self.tokenizer.encode(smiles2, add_special_tokens=False)
        self.assertEqual(tokens2, ["C", "[Cl]"])

        # Ring closures
        smiles3 = "c1ccccc1"
        tokens3 = self.tokenizer.encode(smiles3, add_special_tokens=False)
        self.assertEqual(tokens3, ["c", "1", "c", "c", "c", "c", "c", "1"])

    def test_batch_tokenization(self):
        """Verify batch encoding efficiency."""
        batch_tokens = self.tokenizer.batch_encode(self.smiles_list,
                                                   add_special_tokens=True)
        self.assertEqual(len(batch_tokens), len(self.smiles_list))
        for tokens in batch_tokens:
            self.assertEqual(tokens[0], "<BOS>")
            self.assertEqual(tokens[-1], "<EOS>")

    def test_vocabulary_persistence(self):
        """Verify the save and load round-trip for vocabulary persistence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save vocab
            self.tokenizer.save_vocab(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # Load into a new tokenizer
            new_tokenizer = SmilesTokenizer(level="atom")
            new_tokenizer.load_vocab(tmp_path)

            self.assertEqual(new_tokenizer.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(new_tokenizer.vocab, self.tokenizer.vocab)
            self.assertEqual(new_tokenizer.encode("CCO"),
                             self.tokenizer.encode("CCO"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_reconstruction(self):
        """Ensure encode -> decode faithfully reconstructs original SMILES."""
        for smiles in self.smiles_list:
            ids = self.tokenizer.encode(smiles, return_ids=True)
            reconstructed = self.tokenizer.decode(ids)
            self.assertEqual(reconstructed, smiles)

    def test_dataset_preprocessing(self):
        """Verify dataset preprocessing into PyTorch-ready tensors."""
        try:
            import torch
            max_len = 10
            tensor = self.tokenizer.tokenize_dataset(self.smiles_list,
                                                     max_length=max_len)

            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, (len(self.smiles_list), max_len))
            # Check padding presence
            self.assertIn(self.tokenizer.vocab["<PAD>"], tensor)
        except ImportError:
            self.skipTest("PyTorch is not installed.")

    def test_unknown_token_fallback(self):
        """Verify that unknown atoms are mapped to <UNK>."""
        # Train on basic SMILES
        tokenizer = SmilesTokenizer(level="atom")
        tokenizer.train(["C", "N", "O"])
        
        # 'P' is unknown
        ids = tokenizer.encode("P", return_ids=True)
        # Find index of <UNK>
        unk_id = tokenizer.vocab["<UNK>"]
        # <BOS>, <UNK>, <EOS>
        self.assertEqual(ids[1], unk_id)

    def test_char_level(self):
        """Test character-level tokenization strategy."""
        tokenizer = SmilesTokenizer(level="char")
        tokenizer.train(["CCO"])
        tokens = tokenizer.encode("CCO", add_special_tokens=False)
        self.assertEqual(tokens, ["C", "C", "O"])
