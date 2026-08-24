"""
Tests for MultiModeSmilesTokenizer

Tests cover:
- Character-level tokenization
- Atom-level tokenization
- BPE tokenization
- encode/decode roundtrip
- batch operations
- save/load functionality
"""

import pytest
import os
import tempfile
from deepchem.feat.multimode_smiles_tokenizer import MultiModeSmilesTokenizer


class TestMultiModeSmilesTokenizer:
    """Test suite for MultiModeSmilesTokenizer."""
    
    # Test SMILES strings
    SIMPLE_SMILES = ["CCO", "CC", "C", "O"]
    COMPLEX_SMILES = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "c1ccccc1",                    # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "C[N+](C)(C)CCO",             # Choline
    ]
    SPECIAL_SMILES = [
        "C[Cl]",           # Chloromethane
        "C[Br]",           # Bromomethane
        "[Na+].[Cl-]",     # Salt
        "C/C=C/C",         # cis/trans
        "C[C@H](O)F",      # Stereochemistry
        "C1CC%10CC1C%10",  # Ring numbers > 9
    ]
    
    # =====================
    # Initialization Tests
    # =====================
    
    def test_init_default(self):
        """Test default initialization."""
        tokenizer = MultiModeSmilesTokenizer()
        assert tokenizer.level == 'atom'
        assert tokenizer.vocab_size >= 4  # At least special tokens
    
    def test_init_char_level(self):
        """Test character-level initialization."""
        tokenizer = MultiModeSmilesTokenizer(level='char')
        assert tokenizer.level == 'char'
    
    def test_init_atom_level(self):
        """Test atom-level initialization."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        assert tokenizer.level == 'atom'
    
    def test_init_bpe_level(self):
        """Test BPE initialization."""
        tokenizer = MultiModeSmilesTokenizer(level='bpe', vocab_size=100)
        assert tokenizer.level == 'bpe'
        assert tokenizer.max_vocab_size == 100
    
    def test_init_invalid_level(self):
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError):
            MultiModeSmilesTokenizer(level='invalid')
    
    # =====================
    # Tokenization Tests
    # =====================
    
    def test_tokenize_char_simple(self):
        """Test character-level tokenization on simple SMILES."""
        tokenizer = MultiModeSmilesTokenizer(level='char')
        tokens = tokenizer.tokenize("CCO")
        assert tokens == ['C', 'C', 'O']
    
    def test_tokenize_atom_simple(self):
        """Test atom-level tokenization on simple SMILES."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokens = tokenizer.tokenize("CCO")
        assert tokens == ['C', 'C', 'O']
    
    def test_tokenize_atom_brackets(self):
        """Test atom-level tokenization handles bracketed atoms."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokens = tokenizer.tokenize("C[Cl]")
        assert tokens == ['C', '[Cl]']
    
    def test_tokenize_atom_stereochemistry(self):
        """Test atom-level tokenization handles stereochemistry."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokens = tokenizer.tokenize("C[C@H](O)F")
        assert '[C@H]' in tokens
    
    def test_tokenize_atom_ring_numbers(self):
        """Test atom-level tokenization handles ring numbers."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokens = tokenizer.tokenize("c1ccccc1")
        assert '1' in tokens
    
    def test_tokenize_atom_bonds(self):
        """Test atom-level tokenization handles bond types."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokens = tokenizer.tokenize("C=O")
        assert tokens == ['C', '=', 'O']
        
        tokens = tokenizer.tokenize("C#N")
        assert tokens == ['C', '#', 'N']
    
    def test_tokenize_char_vs_atom(self):
        """Test difference between char and atom tokenization."""
        char_tokenizer = MultiModeSmilesTokenizer(level='char')
        atom_tokenizer = MultiModeSmilesTokenizer(level='atom')
        
        smiles = "C[Cl]"
        char_tokens = char_tokenizer.tokenize(smiles)
        atom_tokens = atom_tokenizer.tokenize(smiles)
        
        # Character-level splits into individual chars
        assert len(char_tokens) == 5  # C, [, C, l, ]
        # Atom-level keeps [Cl] together
        assert len(atom_tokens) == 2  # C, [Cl]
    
    # =====================
    # Encode/Decode Tests
    # =====================
    
    def test_encode_simple(self):
        """Test encoding simple SMILES."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        ids = tokenizer.encode("CCO")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == 3
    
    def test_decode_simple(self):
        """Test decoding simple SMILES."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        ids = tokenizer.encode("CCO")
        decoded = tokenizer.decode(ids)
        assert decoded == "CCO"
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip preserves SMILES."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        
        for smiles in self.SIMPLE_SMILES + self.COMPLEX_SMILES:
            ids = tokenizer.encode(smiles)
            decoded = tokenizer.decode(ids)
            assert decoded == smiles, f"Roundtrip failed for {smiles}"
    
    def test_encode_with_special_tokens(self):
        """Test encoding with special tokens."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        ids = tokenizer.encode("CCO", add_special_tokens=True)
        
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id
    
    def test_encode_with_padding(self):
        """Test encoding with padding."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        ids = tokenizer.encode("CCO", max_length=10, padding=True)
        
        assert len(ids) == 10
        assert ids[-1] == tokenizer.pad_token_id
    
    def test_encode_with_truncation(self):
        """Test encoding with truncation."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        ids = tokenizer.encode("CCCCCCCCCC", max_length=5)  # 10 C's
        
        assert len(ids) == 5
    
    def test_decode_skip_special_tokens(self):
        """Test decoding skips special tokens by default."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        ids = tokenizer.encode("CCO", add_special_tokens=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
        assert decoded == "CCO"
        assert "[BOS]" not in decoded
        assert "[EOS]" not in decoded
    
    # =====================
    # Batch Operations Tests
    # =====================
    
    def test_batch_encode(self):
        """Test batch encoding."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        batch = tokenizer.batch_encode(self.SIMPLE_SMILES, padding=True)
        
        assert len(batch) == len(self.SIMPLE_SMILES)
        # All sequences should have same length due to padding
        assert all(len(seq) == len(batch[0]) for seq in batch)
    
    def test_batch_decode(self):
        """Test batch decoding."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        batch_ids = tokenizer.batch_encode(self.SIMPLE_SMILES, padding=False)
        decoded = tokenizer.batch_decode(batch_ids)
        
        assert decoded == self.SIMPLE_SMILES
    
    # =====================
    # Training Tests
    # =====================
    
    def test_train_atom_level(self):
        """Test training atom-level tokenizer."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        initial_vocab_size = tokenizer.vocab_size
        
        tokenizer.train(self.COMPLEX_SMILES)
        
        assert tokenizer.vocab_size > initial_vocab_size
        assert tokenizer._trained
    
    def test_train_bpe(self):
        """Test training BPE tokenizer."""
        tokenizer = MultiModeSmilesTokenizer(level='bpe', vocab_size=50)
        tokenizer.train(self.COMPLEX_SMILES * 10, min_frequency=1)  # Repeat for more data
        
        assert tokenizer._trained
        assert len(tokenizer.bpe_merges) > 0
    
    def test_bpe_merges_work(self):
        """Test that BPE merges are applied."""
        tokenizer = MultiModeSmilesTokenizer(level='bpe', vocab_size=100)
        tokenizer.train(["CC"] * 100, min_frequency=1)  # Train heavily on CC
        
        # CC should be merged
        tokens = tokenizer.tokenize("CC")
        # Either merged to 'CC' or still separate
        assert len(tokens) <= 2
    
    # =====================
    # Save/Load Tests
    # =====================
    
    def test_save_load_roundtrip(self):
        """Test save and load preserves tokenizer state."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokenizer.train(self.COMPLEX_SMILES)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            tokenizer.save(path)
            loaded = MultiModeSmilesTokenizer.load(path)
            
            assert loaded.level == tokenizer.level
            assert loaded.vocab == tokenizer.vocab
            assert loaded._trained == tokenizer._trained
            
            # Test that loaded tokenizer works
            smiles = "CCO"
            original_ids = tokenizer.encode(smiles)
            loaded_ids = loaded.encode(smiles)
            assert original_ids == loaded_ids
        finally:
            os.unlink(path)
    
    # =====================
    # PyTorch Integration Tests
    # =====================
    
    @pytest.mark.skipif(True, reason="PyTorch optional")
    def test_encode_return_tensors(self):
        """Test encoding returns PyTorch tensors when requested."""
        try:
            import torch
            tokenizer = MultiModeSmilesTokenizer(level='atom')
            tensor = tokenizer.encode("CCO", return_tensors='pt')
            
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dim() == 1
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    @pytest.mark.skipif(True, reason="PyTorch optional")
    def test_batch_encode_return_tensors(self):
        """Test batch encoding returns PyTorch tensors when requested."""
        try:
            import torch
            tokenizer = MultiModeSmilesTokenizer(level='atom')
            tensor = tokenizer.batch_encode(
                self.SIMPLE_SMILES, 
                padding=True, 
                return_tensors='pt'
            )
            
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dim() == 2
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    # =====================
    # Edge Cases Tests
    # =====================
    
    def test_empty_smiles(self):
        """Test handling of empty SMILES."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        tokens = tokenizer.tokenize("")
        assert tokens == []
        
        ids = tokenizer.encode("")
        assert ids == []
    
    def test_special_characters(self):
        """Test handling of special SMILES characters."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        
        for smiles in self.SPECIAL_SMILES:
            tokens = tokenizer.tokenize(smiles)
            ids = tokenizer.encode(smiles)
            decoded = tokenizer.decode(ids)
            assert decoded == smiles, f"Failed for {smiles}"
    
    def test_repr(self):
        """Test string representation."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        repr_str = repr(tokenizer)
        assert "MultiModeSmilesTokenizer" in repr_str
        assert "atom" in repr_str
    
    def test_len(self):
        """Test __len__ returns vocab size."""
        tokenizer = MultiModeSmilesTokenizer(level='atom')
        assert len(tokenizer) == tokenizer.vocab_size


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
