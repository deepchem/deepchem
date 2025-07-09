import numpy as np
from deepchem.feat import Featurizer
from typing import List, Dict, Tuple, Any, Optional

try:
    import torch
    import pysam
    from multiprocessing import Pool, cpu_count
except ImportError:
    pass

BASES = "ACGT"
BASE2IDX = {b: i for i, b in enumerate(BASES)}
N_IDX = 4  # index for 'N'


class _CandidateVariant(object):

    def __init__(self,
                 min_count=2,
                 min_frac=0.01,
                 realign=False,
                 vcf_path=None,
                 offset=0):
        self.min_count = min_count
        self.min_frac = min_frac
        self.realign = realign
        self.vcf_path = vcf_path
        self.offset = offset
        self.variants = self.load_vcf_variants(vcf_path) if vcf_path else {}

    def left_align_indel(self,
                         chrom: str,
                         pos: int,
                         ref: str,
                         alt: str,
                         fasta: 'pysam.FastaFile',
                         max_shift: int = 50) -> Tuple[int, str, str]:
        """
        Left align the indel variant following standard variant normalization practices.

        This function shifts the position of an indel variant to its leftmost possible position
        by checking the reference sequence. It's important for variant normalization to ensure
        consistent representation of the same genetic change.

        Parameters
        ----------
        chrom : str
            Chromosome name where the variant is located
        pos : int
            1-based position of the variant on the chromosome
        ref : str
            Reference allele sequence
        alt : str
            Alternate allele sequence
        fasta : pysam.FastaFile
            FASTA file object providing reference genome sequence
        max_shift : int, optional (default=50)
            Maximum number of bases to try shifting the indel

        Returns
        -------
        Tuple[int, str, str]
            A tuple containing:
            - The left-aligned position (1-based)
            - The normalized reference allele
            - The normalized alternate allele
        """

        if len(ref) == len(alt) or ref[0] != alt[0]:
            return pos, ref, alt
        seq, seq_alt, left = ref, alt, pos
        while len(seq) > 1 and len(seq_alt) > 1 and seq[-1] == seq_alt[-1]:
            seq = seq[:-1]
            seq_alt = seq_alt[:-1]
        while len(seq) > 1 and len(seq_alt) > 1 and seq[0] == seq_alt[0]:
            seq, seq_alt = seq[1:], seq_alt[1:]
            left += 1
        for _ in range(max_shift):
            if left <= 1:
                break
            prev_base = fasta.fetch(chrom, left - 2, left - 1)
            if len(seq) > len(seq_alt):
                if seq[-1] == prev_base:
                    seq = seq[-1] + seq[:-1]
                    left -= 1
                else:
                    break
            else:
                if seq_alt[-1] == prev_base:
                    seq_alt = seq_alt[-1] + seq_alt[:-1]
                    left -= 1
                else:
                    break
        return left, seq, seq_alt

    def seq_to_int(self, seq: str) -> torch.Tensor:
        """
        Convert a DNA sequence string to integer tensor representation.

        This function converts each nucleotide in a DNA sequence to its corresponding
        integer index for use in numerical operations. It handles ACGT bases with
        a special index for any other character (like N).

        Parameters
        ----------
        seq : str
            A DNA sequence string containing nucleotide characters

        Returns
        -------
        torch.Tensor
            A 1D tensor of integers where each number represents a nucleotide:
            - 0 for 'A'
            - 1 for 'C'
            - 2 for 'G'
            - 3 for 'T'
            - 4 for any other character (like 'N')
        """
        return torch.tensor([BASE2IDX.get(c.upper(), N_IDX) for c in seq],
                            dtype=torch.int32)

    def align_smith_waterman_simd(self,
                                  query_sequence: str,
                                  ref_sequence: str,
                                  match_score: int = 2,
                                  mismatch_penalty: int = -1,
                                  gap_open: int = -2,
                                  gap_extend: int = -1) -> str:
        """Perform Smith-Waterman local sequence alignment with SIMD-like optimizations.

        This method implements the Smith-Waterman algorithm for local alignment of DNA sequences
        using PyTorch tensors for efficient computation. It finds the optimal local alignment
        between two sequences by calculating a scoring matrix based on matches, mismatches,
        and gap penalties.

        This implementation uses PyTorch tensors to optimize the calculation of the
        scoring and traceback matrices. The algorithm has O(M*N) time complexity
        where M and N are the lengths of the query and reference sequences.

        The traceback pointer values represent:
        - 0: Stop (no alignment)
        - 1: Diagonal move (match/mismatch)
        - 2: Up move (gap in reference)
        - 3: Left move (gap in query)

        Parameters
        ----------
        query_sequence : str
            The query DNA sequence to align
        ref_sequence : str
            The reference DNA sequence to align against
        match_score : int, optional (default=2)
            Score for matching bases
        mismatch_penalty : int, optional (default=-1)
            Penalty for mismatched bases
        gap_open : int, optional (default=-2)
            Penalty for opening a gap in the alignment
        gap_extend : int, optional (default=-1)
            Penalty for extending an existing gap

        Returns
        -------
        str
            The aligned query sequence with gaps represented as '-'

        """

        query_tensor = self.seq_to_int(query_sequence)
        ref_tensor = self.seq_to_int(ref_sequence)
        M, N = len(query_tensor), len(ref_tensor)
        H = torch.zeros((M + 1, N + 1), dtype=torch.int32)
        E = torch.zeros((M + 1, N + 1), dtype=torch.int32)
        F = torch.zeros((M + 1, N + 1), dtype=torch.int32)
        pointer = torch.zeros((M + 1, N + 1), dtype=torch.int8)
        max_score: float = 0
        max_pos = (0, 0)
        for i in range(1, M + 1):
            match_mask = (query_tensor[i - 1] == ref_tensor)
            sub_scores = match_mask * match_score + (
                ~match_mask) * mismatch_penalty
            for j in range(1, N + 1):
                diag = H[i - 1, j - 1] + sub_scores[j - 1]
                E[i, j] = torch.max(H[i - 1, j] + gap_open,
                                    E[i - 1, j] + gap_extend)
                F[i, j] = torch.max(H[i, j - 1] + gap_open,
                                    F[i, j - 1] + gap_extend)
                vals = torch.tensor([0, diag, E[i, j], F[i, j]])
                best, idx = torch.max(vals, 0)
                H[i, j] = best
                pointer[i, j] = idx
                if best > max_score:
                    max_score = best.item()
                    max_pos = (i, j)
        i, j = max_pos
        aligned_query = []
        while i > 0 and j > 0:
            if pointer[i, j] == 1:
                aligned_query.append(query_sequence[i - 1])
                i -= 1
                j -= 1
            elif pointer[i, j] == 2:
                aligned_query.append(query_sequence[i - 1])
                i -= 1
            elif pointer[i, j] == 3:
                aligned_query.append('-')
                j -= 1
            else:
                break
        return ''.join(aligned_query[::-1])

    def realign_read(self,
                     read: 'pysam.AlignedSegment',
                     ref_seq: str,
                     region_start: int,
                     sw_window: int = 100) -> str:
        """
        Realign a sequencing read to the reference using Smith-Waterman alignment.

        This method performs local realignment of a sequencing read against a reference
        sequence. It extracts a local region from the reference sequence around the read's
        mapping position and uses Smith-Waterman alignment to find the optimal alignment.
        This realignment can improve variant calling accuracy by correctly aligning reads
        around complex variants like indels. The window size determines how much
        reference context is provided for alignment.

        Parameters
        ----------
        read : pysam.AlignedSegment
            The sequencing read to be realigned
        ref_seq : str
            The reference sequence for the entire region
        region_start : int
            The 0-based start position of the reference region in chromosome coordinates
        sw_window : int, optional (default=100)
            Size of the window around the read's position to use for alignment

        Returns
        -------
        str
            The realigned read sequence, or the original sequence if realignment fails

        """
        query_seq = read.query_sequence
        if query_seq is None:
            return ""
        ref_start = max(read.reference_start - region_start - sw_window // 2, 0)
        ref_end = min(ref_start + len(query_seq) + sw_window, len(ref_seq))
        local_ref = ref_seq[ref_start:ref_end]
        if not local_ref:
            return query_seq
        return self.align_smith_waterman_simd(query_seq, local_ref)

    def count_alleles(
        self,
        reads: List['pysam.AlignedSegment'],
        ref_seq: str,
        region_start: int,
        realign_reads: Optional[Dict[Tuple[str, int], str]] = None
    ) -> List[Dict[str, int]]:
        """
        Count alleles at each position in the reference sequence from aligned reads.

        This method counts the occurrences of each nucleotide base at each position
        in the reference sequence by parsing the CIGAR string of each read and
        tracking the alignment between reference and query positions.

        The method handles different CIGAR operations:
        - 0 (M): Match or mismatch - both reference and query positions advance
        - 1 (I): Insertion - only query position advances
        - 2 (D): Deletion - only reference position advances
        - 4 (S): Soft clip - only query position advances
        - 5 (H): Hard clip - positions unchanged

        Unmapped and duplicate reads are skipped.

        Parameters
        ----------
        reads : List[pysam.AlignedSegment]
            List of aligned reads from a BAM/SAM file
        ref_seq : str
            Reference sequence for the region being analyzed
        region_start : int
            0-based start position of the region in chromosome coordinates
        realign_reads : Optional[Dict[Tuple[str, int], str]], default=None
            Dictionary mapping (read_name, reference_start) to realigned sequences.
            If provided, these sequences will be used instead of the original ones.

        Returns
        -------
        List[Dict[str, int]]
            A list of dictionaries, one for each reference position, where each dictionary
            maps nucleotide bases to their counts at that position
        """
        counts: List[Dict[str, int]] = [{} for _ in range(len(ref_seq))]
        for read in reads:
            if read.is_unmapped or read.is_duplicate:
                continue
            read_name = read.query_name if read.query_name is not None else ""
            seq = None
            if realign_reads is not None and (
                    read_name, read.reference_start) in realign_reads:
                seq = realign_reads.get((read_name, read.reference_start))
            else:
                seq = read.query_sequence

            if seq is None:
                continue

            ref_pos, query_pos = read.reference_start, 0
            cigartuples = read.cigartuples or []

            for cigartupe, length in cigartuples:
                if cigartupe == 0:  # Match/mismatch
                    for i in range(length):
                        pos = ref_pos + i - region_start
                        if 0 <= pos < len(ref_seq) and query_pos + i < len(seq):
                            base = seq[query_pos + i]
                            counts[pos][base] = counts[pos].get(base, 0) + 1
                    ref_pos += length
                    query_pos += length
                elif cigartupe == 1:  # Insertion
                    query_pos += length
                elif cigartupe == 2:  # Deletion
                    ref_pos += length
                elif cigartupe == 4:  # Soft clip
                    query_pos += length
                elif cigartupe == 5:  # Hard clip
                    pass
                else:
                    ref_pos += length
        return counts

    def detect_candidates(
            self,
            counts: List[Dict[str, int]],
            ref_seq: str,
            region_start: int,
            min_count: int = 2,
            min_frac: float = 0.01) -> List[Tuple[int, str, str, int, int]]:
        """Detect candidate genetic variants by analyzing allele counts.

        This method identifies potential variants by comparing the observed alleles
        at each position to the reference sequence. It filters variants based on
        minimum count and minimum frequency thresholds to reduce false positives.

        This method is typically used after counting alleles to identify
        positions where there may be genetic variants worth investigating.

        Parameters
        ----------
        counts : List[Dict[str, int]]
            List of dictionaries containing base counts at each position,
            where each dictionary maps nucleotide bases to their counts
        ref_seq : str
            Reference sequence for the region being analyzed
        region_start : int
            0-based start position of the region in chromosome coordinates
        min_count : int, optional (default=2)
            Minimum number of reads supporting the alternate allele
        min_frac : float, optional (default=0.01)
            Minimum fraction of reads supporting the alternate allele

        Returns
        -------
        List[Tuple[int, str, str, int, int]]
            A list of tuples for each candidate variant, where each tuple contains:
            - Position index in the reference sequence
            - Reference base at that position
            - Alternate base observed in the reads
            - Count of reads supporting the alternate allele
            - Total number of reads covering the position

        """
        candidates = []
        for i, cnt in enumerate(counts):
            n_total = sum(cnt.values())
            if n_total == 0:
                continue
            ref_base = ref_seq[i].upper()
            for base, count in cnt.items():
                if base != ref_base and count >= min_count and count / n_total >= min_frac:
                    candidates.append((i, ref_base, base, count, n_total))
        return candidates

    def get_contigs_from_fasta(self, fasta_path: str) -> List[Tuple[str, int]]:
        """
        Extract contig information from a FASTA reference file.

        This method reads a FASTA file and returns information about all contigs
        (chromosomes or scaffolds) it contains, including their names and lengths.

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file containing reference genome sequences

        Returns
        -------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains:
            - The contig name (str)
            - The contig length in base pairs (int)
        """
        fasta = pysam.FastaFile(fasta_path)
        contigs = [
            (ctg, fasta.get_reference_length(ctg)) for ctg in fasta.references
        ]
        fasta.close()
        return contigs

    def region_generator(self,
                         contigs: List[Tuple[str, int]],
                         window_size: int = 100000):
        """
        Generate genomic regions by dividing contigs into fixed-size windows.

        This method takes a list of contigs (chromosomes) and their lengths, and
        divides each contig into non-overlapping windows of a specified size.
        It yields each region as a tuple containing the contig name and the
        start and end positions.

        Parameters
        ----------
        contigs : List[Tuple[str, int]]
            A list of tuples, where each tuple contains:
            - The contig name (str)
            - The contig length in base pairs (int)
        window_size : int, optional (default=100000)
            Size of each window in base pairs

        Yields
        ------
        Tuple[str, int, int]
            A tuple containing:
            - Contig name (str)
            - Start position (0-based, inclusive)
            - End position (0-based, exclusive)
        """
        for ctg, length in contigs:
            for start in range(0, length, window_size):
                end = min(start + window_size, length)
                yield (ctg, start, end)

    def load_vcf_variants(
            self,
            vcf_path: str) -> Dict[Tuple[str, int, str, str], Optional[Tuple]]:
        """Load variant information from a VCF file.

        This method parses a VCF file and extracts variant information, including
        chromosome, position, reference allele, alternate allele, and genotype.

        Parameters
        ----------
        vcf_path : str
            Path to the VCF file containing variant information

        Returns
        -------
        Dict[Tuple[str, int, str, str], Optional[Tuple]]
            A dictionary mapping variant keys to genotype information:
            - Keys are tuples of (chromosome, position, reference_allele, alternate_allele)
            - Values are genotype (GT) values from the VCF, or None if not available

        """
        vcf = pysam.VariantFile(vcf_path)
        variants: Dict[Tuple[str, int, str, str], Optional[Tuple[Any,
                                                                 ...]]] = {}
        try:
            for rec in vcf.fetch():
                chrom = rec.chrom
                pos = rec.pos
                ref = rec.ref
                if chrom is None or ref is None or rec.alts is None:
                    continue
                chrom_str = str(chrom)
                ref_str = str(ref)
                for alt_idx, alt in enumerate(rec.alts):
                    if alt is None:
                        continue
                    alt_str = str(alt)
                    key = (chrom_str, pos, ref_str, alt_str)
                    gt = None
                    if rec.samples:
                        sample = list(rec.samples.values())[0]
                        gt = sample.get('GT', None)
                    variants[key] = gt
        except Exception:
            pass

        return variants

    def label_variant(self, chrom: str, pos: int, ref: str, alt: str) -> int:
        """
        Determine the genotype label for a specific variant.

        This method looks up a variant in the loaded VCF data and determines
        its genotype classification (homozygous reference, heterozygous,
        or homozygous alternate).

        Parameters
        ----------
        chrom : str
            Chromosome name where the variant is located
        pos : int
            Position coordinate of the variant
        ref : str
            Reference allele
        alt : str
            Alternate allele

        Returns
        -------
        int
            Genotype label:
            - 0: Homozygous reference or variant not in truth set
            - 1: Heterozygous variant
            - 2: Homozygous alternate variant
        """
        pos = int(pos) + self.offset
        label = 0
        gt = self.variants.get((chrom, pos, ref, alt))
        if gt is not None:
            gt_clean = [x for x in gt if x is not None]
            if len(gt_clean) == 0:
                pass
            elif any(x == 1 for x in gt_clean) and any(
                    x == 0 for x in gt_clean):
                label = 1
            elif all(x > 0 for x in gt_clean):
                label = 2
        return label

    def process_region(self, chrom: str, start: int, end: int, bam_path: str,
                       fasta_path: str, min_count: int, min_frac: float,
                       realign: bool, label: bool) -> List[List[Any]]:
        """Process a genomic region to identify variant candidates.

        This method handles the complete variant calling pipeline for a specific genomic region:
        1. Loads reference sequence and aligned reads
        2. Optionally realigns reads to improve alignment quality
        3. Counts alleles at each position
        4. Identifies candidate variants that meet specified criteria
        5. Left-aligns indels to ensure consistent representation
        6. Optionally labels variants based on truth set

        Parameters
        ----------
        chrom : str
            Chromosome or contig name
        start : int
            0-based start position of the region
        end : int
            0-based end position of the region (exclusive)
        bam_path : str
            Path to BAM file containing aligned reads
        fasta_path : str
            Path to FASTA file containing reference genome
        min_count : int
            Minimum number of reads supporting an alternate allele
        min_frac : float
            Minimum fraction of reads supporting an alternate allele
        realign : bool
            Whether to perform read realignment
        label : bool
            Whether to label variants using truth set

        Returns
        -------
        List[List[Any]]
            List of candidate variants, where each variant is represented as a list:
            - [0]: Chromosome name (str)
            - [1]: 0-based position (int)
            - [2]: Reference allele (str)
            - [3]: Alternate allele (str)
            - [4]: Count of reads supporting the alternate allele (int)
            - [5]: Total reads covering the position (int)
            - [6]: (Optional) Variant label (int, present if label=True)
        """
        fasta = pysam.FastaFile(fasta_path)
        bam = pysam.AlignmentFile(bam_path)
        ref_seq = fasta.fetch(chrom, start, end)
        reads = list(bam.fetch(chrom, start, end))

        realign_reads: Optional[Dict[Tuple[str, int], str]] = None
        if realign:
            realign_reads = {}
            for read in reads:
                if read.query_sequence is None:
                    continue
                read_name = read.query_name if read.query_name is not None else ""
                realigned_seq = self.realign_read(read, ref_seq, start)
                key = (read_name, read.reference_start)
                realign_reads[key] = realigned_seq

        counts = self.count_alleles(reads, ref_seq, start, realign_reads)
        candidates = self.detect_candidates(counts, ref_seq, start, min_count,
                                            min_frac)

        records = []
        for pos, ref_base, alt_base, count, total in candidates:
            pos1 = start + pos
            left, lref, lalt = self.left_align_indel(chrom, pos1 + 1, ref_base,
                                                     alt_base, fasta)
            row = [chrom, left - 1, lref, lalt, count, total]
            if label:
                lab = self.label_variant(chrom, left - 1, lref, lalt)
                row.append(lab)
            records.append(row)
        bam.close()
        fasta.close()
        return records


class CandidateVariantFeaturizer(Featurizer):
    """Featurizer for generating candidate genomic variant windows from sequencing data.

    This featurizer processes BAM and reference FASTA files to identify
    candidate variants. It provides haplotype awareness through read realignment
    using the Smith-Waterman algorithm. The pipeline includes:
    1. Processing BAM/FASTA files to generate allele counts
    2. Detecting candidate variants based on allele frequencies
    3. Performing optional read realignment around variants
    4. Left-aligning indels for consistent representation
    5. Optionally labeling variants using a truth set

    Parameters
    ----------
    window_size : int, optional (default=100000)
        Size of genomic windows for processing in base pairs
    min_count : int, optional (default=2)
        Minimum number of reads supporting an alternate allele
    min_frac : float, optional (default=0.01)
        Minimum fraction of reads that must support an alternate allele
    realign : bool, optional (default=False)
        Whether to perform read realignment using Smith-Waterman
    vcf_path : str, optional (default=None)
        Path to truth VCF file for labeling variants
    offset : int, optional (default=0)
        Position offset to apply when matching variants to truth set
    multiprocessing : bool, optional (default=False)
        Whether to process regions in parallel
    threads : int, optional (default=None)
        Number of threads for parallel processing (defaults to CPU count)

    Examples
    --------
    >>> from deepchem.feat import CandidateVariantFeaturizer
    >>> bamfile_path = 'deepchem/data/tests/example.bam'
    >>> reference_path = 'deepchem/data/tests/sample.fa'
    >>> featurizer = CandidateVariantFeaturizer()
    >>> datapoint = (bamfile_path, reference_path)
    >>> features = featurizer.featurize([datapoint])

    Note
    ----
    This class requires Pysam, DGL and Pytorch to be installed. Pysam can be
    used with Linux or MacOS X. To use Pysam on Windows, use Windows
    Subsystem for Linux(WSL).

    """

    def __init__(self,
                 window_size: int = 100000,
                 min_count: int = 2,
                 min_frac: float = 0.01,
                 realign: bool = False,
                 vcf_path: Optional[str] = None,
                 offset: int = 0,
                 multiprocessing: bool = False,
                 threads: Optional[int] = None) -> None:
        self.window_size = window_size
        self.min_count = min_count
        self.min_frac = min_frac
        self.realign = realign
        self.vcf_path = vcf_path
        self.offset = offset
        self.multiprocessing = multiprocessing
        self.threads = threads or cpu_count()

    def _featurize(self, datapoint):
        """Process a BAM/FASTA pair to extract candidate variants.

        Parameters
        ----------
        datapoint : Tuple[str, str]
            A tuple containing (bam_path, fasta_path) where:
            - bam_path is the path to a BAM file with aligned reads
            - fasta_path is the path to a FASTA file with the reference genome

        Returns
        -------
        np.ndarray
            Array of candidate variants, where each variant is represented as:
            - [0]: Chromosome name (str)
            - [1]: 0-based position (int)
            - [2]: Reference allele (str)
            - [3]: Alternate allele (str)
            - [4]: Count of reads supporting the alternate allele (int)
            - [5]: Total reads covering the position (int)
            - [6]: (Optional) Variant label (int, present if vcf_path is provided)
        """

        bam_path, fasta_path = datapoint
        windower = _CandidateVariant(min_count=self.min_count,
                                     min_frac=self.min_frac,
                                     realign=self.realign,
                                     vcf_path=self.vcf_path,
                                     offset=self.offset)
        contigs = windower.get_contigs_from_fasta(fasta_path)
        regions = list(windower.region_generator(contigs, self.window_size))
        label = self.vcf_path is not None

        args = [(chrom, start, end, bam_path, fasta_path, self.min_count,
                 self.min_frac, self.realign, label)
                for (chrom, start, end) in regions]

        if self.multiprocessing:
            with Pool(self.threads) as pool:
                all_records = pool.starmap(windower.process_region, args)
        else:
            all_records = [windower.process_region(*arg) for arg in args]

        out = []
        for recs in all_records:
            out.extend(recs)

        return np.array(out, dtype=object)
