from torch.utils.data import Dataset


class FASTADataset(Dataset):
    """
    Simple dataset loader for DNA sequences stored in FASTA format.
    """

    def __init__(self, fasta_file, seq_length=128):

        self.sequences = []
        self.seq_length = seq_length

        with open(fasta_file, "r") as f:

            current_seq = ""

            for line in f:

                line = line.strip()

                if line.startswith(">"):

                    if current_seq:
                        self._split_sequence(current_seq)

                    current_seq = ""

                else:
                    current_seq += line

            if current_seq:
                self._split_sequence(current_seq)

    def _split_sequence(self, sequence):

        for i in range(0, len(sequence) - self.seq_length, self.seq_length):

            fragment = sequence[i : i + self.seq_length]

            self.sequences.append(fragment)

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        return self.sequences[idx]