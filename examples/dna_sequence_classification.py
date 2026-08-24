# Example for DNA Sequence Classification using simple tokenization

from deepchem.utils.dna_tokenizer import tokenize_dna, kmer_tokenize

# Sample DNA sequences
sequences = ["ATGCGTAA", "CGTATGCA"]

# Tokenize sequences
tokenized_sequences = [tokenize_dna(seq) for seq in sequences]

print("Tokenized Sequences:")
print(tokenized_sequences)

# k-mer tokenization
kmer_sequences = [kmer_tokenize(seq, k=3) for seq in sequences]

print("\nK-mer Tokenized Sequences:")
print(kmer_sequences)

# Dummy labels (for illustration)
labels = [1, 0]

print("\nLabels:")
print(labels)

# NOTE:
# This is a simple demo of preprocessing steps
# for DNA sequence modeling in DeepChem 
# Only to Illustrate pipeline