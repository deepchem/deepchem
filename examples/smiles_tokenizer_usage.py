"""
Example Usage for DeepChem SmilesTokenizer
------------------------------------------
This script demonstrates the production-level features of the SmilesTokenizer,
including training, batch preprocessing for PyTorch, and persistence.
"""

import os
import sys

# Add the current directory to path if needed to find deepchem
sys.path.append(os.getcwd())

# Mock common DeepChem and RDKit modules if they are missing locally.
# This ensures the example runs even if the host is missing scientific dependencies.
import types

def mock_dependencies():
    for mname in ['rdkit', 'rdkit.Chem', 'rdkit.Chem.rdmolops', 'rdkit.Chem.AllChem']:
        if mname not in sys.modules:
            sys.modules[mname] = types.ModuleType(mname)

mock_dependencies()

from deepchem.feat.smiles_tokenizer import SmilesTokenizer

def main():
    print("--- 1. Initialization and Training ---")
    # Initialize with atom-level parsing (production default)
    tokenizer = SmilesTokenizer(level="atom")
    
    # Example corpus
    smiles_corpus = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic Acid
    ]
    
    # Train the vocabulary on the dataset
    tokenizer.train(smiles_corpus)
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    # Example tokens: <PAD>, <UNK>, <BOS>, <EOS>, C, (, =, O, ), r, k, etc.
    print(f"Sample mapping: 'C' -> {tokenizer.vocab.get('C')}")

    print("\n--- 2. Dataset Preprocessing (PyTorch) ---")
    # Preprocess a list of molecules for a sequence model (e.g., Transformer, LSTM)
    # We set a max length and it returns a padded PyTorch tensor.
    try:
        import torch
        # tokenize_dataset handles <BOS>, <EOS>, truncation, and <PAD> padding.
        tensor = tokenizer.tokenize_dataset(smiles_corpus, max_length=15)
        print("Dataset Tensor Shape:", tensor.shape)
        # Sequence for Aspirin: <BOS>, C, C, (, =, O, ), ..., <EOS>, <PAD>...
        print("First molecule tensor sample:", tensor[0, :8])
    except ImportError:
        print("PyTorch not installed. Returning list of IDs instead.")
        batch_ids = tokenizer.batch_encode(smiles_corpus, return_ids=True)
        print("Batch IDs length:", len(batch_ids))

    print("\n--- 3. Reversibility (Encode/Decode) ---")
    # Check if we can reconstruct the molecule
    mol = "C[C@H](O)C"  # Isopropanol with stereochemistry
    if "[C@H]" not in tokenizer.vocab:
        # In literal use, you'd train on your specific molecules
        tokenizer.train([mol])
    
    ids = tokenizer.encode(mol, return_ids=True)
    reconstructed = tokenizer.decode(ids)
    print(f"Original: {mol} -> Tokens: {tokenizer.encode(mol, add_special_tokens=False)}")
    print(f"Reconstructed: {reconstructed}")
    assert mol == reconstructed

    print("\n--- 4. Persistence ---")
    # Save the vocabulary to disk for future inference or shared use
    vocab_path = "smiles_vocab.json"
    tokenizer.save_vocab(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    # Load into a new instance
    new_tokenizer = SmilesTokenizer(level="atom")
    new_tokenizer.load_vocab(vocab_path)
    print(f"Loaded vocab size: {new_tokenizer.vocab_size}")
    
    # Clean up
    if os.path.exists(vocab_path):
        os.remove(vocab_path)

if __name__ == "__main__":
    main()
