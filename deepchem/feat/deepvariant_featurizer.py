import numpy as np
from collections import defaultdict
from deepchem.feat import Featurizer
from typing import List, Dict, Tuple, Any


class _Realigner(object):

    def left_align_indel(self, seq: str, pos: int,
                         indel: str) -> Tuple[int, str]:
        """
        Left align an indel by shifting it to the left as
        much as possible.

        Parameters
        ----------

        seq : str
            Reference sequence as a string.
        pos : int
            Position of the indel.
        indel : str
            Indel sequence (e.g., "+2AT" or "-2").

        Returns
        -------

        Tuple[int, str]
            New position and indel string.

        """
        if indel.startswith('+'):
            indel_seq = indel[2:]
            while pos > 0 and seq[pos - 1] == indel_seq[-1]:
                pos -= 1
                indel_seq = seq[pos] + indel_seq[:-1]
            return pos, f"+{len(indel_seq)}{indel_seq}"
        elif indel.startswith('-'):
            del_len = int(indel[1:])
            while pos > 0 and seq[pos - 1] == seq[pos + del_len - 1]:
                pos -= 1
            return pos, f"-{del_len}"
        return pos, indel

    def decode_one_hot(self,
                       one_hot_vector: List[np.ndarray],
                       charset: List[str] = ["A", "C", "T", "G", "N"]) -> str:
        """
        Decode a one-hot encoded sequence into a string of
        nucleotides.

        Parameters
        ----------

        one_hot_vector : List[np.ndarray]
            List of one-hot encoded vectors.
        charset : Optional[List[str]]
            List of characters corresponding to the encoding.
            Default is ["A", "C", "T", "G", "N"].

        Returns
        -------

        str
            Decoded sequence as a string.

        """
        decoded_seq = []
        for vector in one_hot_vector:
            idx = np.argmax(vector)
            decoded_seq.append(charset[idx])
        return ''.join(decoded_seq)

    def process_pileup(
            self, pileup_info: List[Dict[str,
                                         Any]], reference_seq_dict: Dict[str,
                                                                         str],
            allele_counts: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
        """
        Process pileup information to extract allele counts.

        Parameters
        ----------

        pileup_info : List[Dict[str, Any]]
            List of dictionaries containing pileup information.
        reference_seq_dict : Dict[str, str]
            Dictionary with chromosome names as keys and reference
            sequences as values.
        allele_counts : Dict[Tuple[str, int], Dict[str, Any]]
            Dictionary to store allele counts.

        """
        for pileupcolumn in pileup_info:
            ref_base = reference_seq_dict[pileupcolumn['name']][
                pileupcolumn['pos']]
            allele_count: Dict[str, Any] = {
                "reference_base": ref_base,
                "read_alleles": defaultdict(int),
                "coverage": 0
            }
            for read_seq in pileupcolumn['reads']:
                if read_seq[2]:
                    base = '-'
                    print(allele_count)
                    allele_count["read_alleles"][base] += 1
                    allele_count["coverage"] += 1
                elif read_seq[3]:
                    base = 'N'
                    allele_count["read_alleles"][base] += 1
                    allele_count["coverage"] += 1
                else:
                    indel = ""
                    if read_seq[4] > 0:
                        indel_seq = read_seq[0][read_seq[2]:read_seq[2] +
                                                read_seq[4]]
                        indel = f"+{read_seq[4]}{indel_seq}"
                        new_pos, indel = self.left_align_indel(
                            reference_seq_dict[pileupcolumn['name']]
                            [0:pileupcolumn['pos'] + read_seq[4] + 1],
                            pileupcolumn['pos'], indel)
                    elif read_seq[4] < 0:
                        indel = f"{read_seq[4]}"
                        new_pos, indel = self.left_align_indel(
                            reference_seq_dict[pileupcolumn['name']]
                            [0:pileupcolumn['pos'] + read_seq[4] + 1],
                            pileupcolumn['pos'], indel)

                    base = read_seq[0][read_seq[1]]

                    if base in {'A', 'C', 'G', 'T'}:
                        if indel:
                            allele_count["read_alleles"][base + indel] += 1
                            allele_count["coverage"] += 1
                        else:
                            allele_count["read_alleles"][base] += 1
                            allele_count["coverage"] += 1
            pos = pileupcolumn['pos'] + 1
            allele_counts[(pileupcolumn['name'], pos)] = allele_count

    def generate_pileup_and_reads(
        self, bamfile_path: str, reference_path: str
    ) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Any]]:
        """
        Generate pileup and reads from BAM and reference FASTA files.

        Parameters
        ----------

        bamfile_path : str
            Path to the BAM file.
        reference_path : str
            Path to the reference FASTA file.

        Returns
        -------

        Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Any]]
            Dictionary of allele counts and list of reads.

        """
        from deepchem.data import FASTALoader, BAMLoader

        bam_loader = BAMLoader(get_pileup=True)
        bam_dataset = bam_loader.create_dataset(bamfile_path)

        fasta_loader = FASTALoader(None, False, False)
        fasta_dataset = fasta_loader.create_dataset(reference_path)

        allele_counts: Dict[Tuple[str, int], Dict[str, Any]] = {}
        reads = []

        one_hot_encoded_sequences = fasta_dataset.X
        decoded_sequences: List[str] = []

        # Convert the one-hot encoded sequences to strings
        for seq in one_hot_encoded_sequences:
            decoded_seq = self.decode_one_hot(seq)
            decoded_sequences.append(decoded_seq)

        # Map the sequences to chrom names
        chrom_names = ["chr1", "chr2"]

        reference_seq_dict: Dict[str, str] = {
            chrom_names[i]: seq for i, seq in enumerate(decoded_sequences)
        }

        for x, y, w, ids in bam_dataset.itersamples():
            chrom = x[3]  # Reference name

            pileup_info = x[7] if len(x) > 7 else None
            for pileupcolumn in pileup_info:
                for read_seq in pileupcolumn['reads']:
                    reads.append(read_seq)

            if chrom not in reference_seq_dict:
                continue

            if pileup_info:
                self.process_pileup(pileup_info, reference_seq_dict,
                                    allele_counts)

        return allele_counts, reads

    def update_counts(self, count: int, start: int, end: int,
                      window_counts: Dict[int, int]) -> None:
        """
        Update counts in a window.

        Parameters
        ----------

        count : int
            Count to add.
        start : int
            Start position of the window.
        end : int
            End position of the window.
        window_counts : Dict[int, int]
            Dictionary to store window counts.

        """
        for pos in range(start, end):
            window_counts[pos] += count

    def select_candidate_regions(
        self, allele_counts: Dict[Tuple[str, int], Dict[str, Any]]
    ) -> List[Tuple[str, int, int, int]]:
        """
        Select candidate regions based on allele counts.

        Parameters
        ----------

        allele_counts : Dict[Tuple[str, int], Dict[str, Any]]
            Dictionary of allele counts.

        Returns
        -------

        List[Tuple[str, int, int, int]]
            List of candidate regions.

        """
        window_counts: Dict[int, int] = defaultdict(int)
        chrom_positions = defaultdict(list)

        for (chrom, pos), allele_count in allele_counts.items():
            for allele, count in allele_count["read_alleles"].items():
                if allele != allele_count["reference_base"]:
                    allele_type = ""
                    if allele.startswith("+") or allele.startswith("-"):
                        allele_length = int(allele[1:])
                        allele_type = "INSERTION" if allele.startswith(
                            "+") else "DELETION"
                    elif allele == "N":
                        allele_type = "SOFT_CLIP"
                        allele_length = 1
                    else:
                        allele_type = "SUBSTITUTION"
                        allele_length = 1

                    start, end = 0, 0

                    if allele_type == "SUBSTITUTION":
                        start = pos
                        end = pos + 1
                    elif allele_type in ["SOFT_CLIP", "INSERTION"]:
                        start = pos + 1 - (allele_length - 1)
                        end = pos + allele_length
                    elif allele_type == "DELETION":
                        start = pos + 1
                        end = pos + allele_length

                    self.update_counts(count, start, end, window_counts)
                    chrom_positions[chrom].append((start, end))

        candidate_regions = []
        for chrom, positions in chrom_positions.items():
            positions.sort()
            if positions:
                start, end = positions[0]
                current_count = sum(
                    window_counts[pos] for pos in range(start, end))

                for (next_start, next_end) in positions[1:]:
                    if next_start <= end + 1:
                        end = max(end, next_end)
                        current_count += sum(
                            window_counts[pos]
                            for pos in range(next_start, next_end))
                    else:
                        candidate_regions.append(
                            (chrom, start, end, current_count))
                        start, end = next_start, next_end
                        current_count = sum(
                            window_counts[pos] for pos in range(start, end))

                candidate_regions.append((chrom, start, end, current_count))

        return candidate_regions


class RealignerFeaturizer(Featurizer):
    """
    Realigns reads and generates candidate regions for variant calling.

    More features can be added to this class in the future.

    """

    def __init__(self):
        self.realigner = _Realigner()

    def _featurize(self, datapoint):
        """
        Featurizes a datapoint by generating candidate regions and reads.

        Args:
            datapoint (Tuple[str, str]): A tuple containing two strings
            representing allele counts and reads.

        Returns:
            Tuple[List[Tuple[str, int, int, int]], List[Any]]: A tuple
            containing the candidate regions and reads.

        """
        allele_counts, reads = self.realigner.generate_pileup_and_reads(
            datapoint[0], datapoint[1])
        candidate_regions = self.realigner.select_candidate_regions(
            allele_counts)
        return candidate_regions, reads
