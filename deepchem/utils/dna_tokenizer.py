def tokenize_dna(sequence):
    return list(sequence)

def kmer_tokenize(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]