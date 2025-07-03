import numpy as np
from deepchem.feat import Featurizer
from deepchem.data import NumpyDataset

try:
    import pysam
except ImportError:
    pass


class _PileupImage(object):
    """
    Generates pileup image representations of genomic variants from BAM and
    FASTA files.

    This class creates a multi-channel image representation of sequence
    alignments around a genomic variant position, similar to how genome
    browsers display read alignments but encoded as numerical arrays.

    The resulting image is a 3D tensor where:
    - Dimension 1 (channels): Different types of sequencing information
    - Dimension 2 (height): Individual sequencing reads (rows)
    - Dimension 3 (width): Genomic positions (columns) in the analysis window

    The image contains channels for:
    1. Base identity: Nucleotide type encoded as intensity
    2. Base quality scores: Phred-scaled quality normalized to [0,1] range
    3. Mapping quality: How confidently the read maps to this location [0,1]
    4. Read strand: Forward =1.0, reverse =0.0 (detects strand bias)
    5. Alternate allele match: 1.0 if base matches variant allele, else 0.0
    6. Reference allele match: 1.0 if base matches reference allele, else 0.0

    """

    def __init__(self,
                 window: int = 221,
                 height: int = 100,
                 channels: int = 6) -> None:
        """
        Initialize the pileup image generator.

        Parameters
        ----------
        window : int, optional (default=221)
            Width of the pileup image in bases, centered on the variant
            position
        height : int, optional (default=100)
            Height of the pileup image in reads, including one row for
            the reference
        channels : int, optional (default=6)
            Number of channels in the pileup image
        """
        self.window = window
        self.height = height
        self.channels = channels

    def base_to_intensity(self, base: str) -> float:
        """
        Convert a nucleotide base to a normalized intensity value.

        This encoding allows the four DNA bases to be distinguished in the base
        identity channel while maintaining numerical relationships that may be
        meaningful for machine learning (e.g., purines A,G vs pyrimidines C,T).

        Parameters
        ----------
        base : str
            A single nucleotide character (A, C, G, T, or other).
            Case-insensitive.

        Returns
        -------
        float
            Normalized intensity value in range [0,1]:
            - A (Adenine): 0.25
            - C (Cytosine): 0.50
            - G (Guanine): 0.75
            - T (Thymine): 1.00
            - Other bases (N, gaps, etc): 0.00

        """
        if base == "A":
            return 0.25
        elif base == "C":
            return 0.5
        elif base == "G":
            return 0.75
        elif base == "T":
            return 1.0
        else:
            return 0.0

    def make_image(self, bam_path: str, fasta_path: str, chrom: str, pos: int,
                   ref: str, alt: str) -> np.ndarray:
        """
        Generate a pileup image for a genomic variant.

        This method creates a multi-channel image representation of
        aligned reads around a variant position. The process involves:
        1. Extracting reference sequence for the analysis window
        2. Fetching all reads overlapping the window from the BAM file
        3. Sorting reads by quality (best reads first)
        4. Encoding each read's bases and metadata into the 6-channel image
        5. Adding the reference sequence as the bottom row for comparison

        The resulting image can reveal patterns such as:
        - Strand bias (variants appearing mostly on one strand)
        - Quality patterns (low-quality bases concentrated around variants)
        - Allele frequency (proportion of reads supporting each allele)
        - Mapping artifacts (poor mapping quality around certain regions)

        Parameters
        ----------
        bam_path : str
            Path to the indexed BAM file containing aligned sequencing reads.
            BAM file must be sorted and indexed (.bai file present).
        fasta_path : str
            Path to the indexed FASTA file containing the reference genome.
            FASTA file must be indexed (.fai file present).
        chrom : str
            Chromosome or contig name where the variant is located.
            Must match chromosome naming in both BAM and FASTA files.
        pos : int
            1-based genomic position of the variant on the chromosome.
            This position will be centered in the analysis window.
        ref : str
            Reference allele sequence at the variant position.
            Used to determine reference matches in channel 5.
        alt : str
            Alternate (variant) allele sequence.
            Used to determine alternate matches in channel 4.

        Returns
        -------
        np.ndarray
            A 3D tensor with shape (channels, height, window) representing
            the pileup image. The channels are:
            - Channel 0: Base identity (A=0.25, C=0.5, G=0.75, T=1.0, other=0)
            - Channel 1: Base quality score (normalized to 0-1 range)
            - Channel 2: Mapping quality (normalized to 0-1 range)
            - Channel 3: Read strand (1.0=forward, 0.0=reverse)
            - Channel 4: Match to alternate allele (1.0=match, 0.0=no match)
            - Channel 5: Match to reference allele (1.0=match, 0.0=no match)
        """
        # Calculate genomic window boundaries
        start = pos - self.window // 2
        end = pos + self.window // 2 + 1

        # Open reference and alignment files
        fasta = pysam.FastaFile(fasta_path)
        bam = pysam.AlignmentFile(bam_path)

        # Extract reference sequence, handling boundary cases
        fetch_start = max(0, start)
        ref_seq = fasta.fetch(chrom, fetch_start, end)

        # Pad with 'N' if window extends before chromosome start
        left_pad = "N" * (0 - start) if start < 0 else ""
        ref_seq = left_pad + ref_seq

        # Pad with 'N' if window extends beyond chromosome end
        if len(ref_seq) < self.window:
            ref_seq += "N" * (self.window - len(ref_seq))

        # Fetch and sort reads by quality (best first)
        reads = list(bam.fetch(chrom, fetch_start, end))
        reads = sorted(
            reads,
            key=lambda r:
            (-r.mapping_quality, r.is_reverse, r.is_secondary, r.query_name))

        # Initialize the pileup image tensor
        pile = np.zeros((self.channels, self.height, self.window),
                        dtype=np.float32)

        # Fill reference row
        for col in range(self.window):
            base = ref_seq[col]
            pile[0, self.height - 1, col] = self.base_to_intensity(base)
            pile[1, self.height - 1, col] = 1.0
            pile[2, self.height - 1, col] = 1.0
            pile[3, self.height - 1, col] = 1.0
            pile[4, self.height - 1,
                 col] = 1.0 if base.upper() == alt.upper() else 0.0
            pile[5, self.height - 1, col] = 1.0

        # Process reads
        for row, read in enumerate(reads[:self.height - 1]):
            seq = read.query_sequence
            if seq is None:
                continue

            # Get base qualities
            read_quals = read.query_qualities
            quals = read_quals if read_quals is not None else [20] * len(seq)
            is_reverse = read.is_reverse
            mq = min(read.mapping_quality, 60) / 60.0

            # Process each aligned position in the read
            for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                if rpos is None or not (start <= rpos < end):
                    continue
                # Convert to image column coordinate
                col = rpos - start

                # Get base, handling gaps/deletions
                if qpos is None or qpos >= len(seq):
                    base = "N"
                else:
                    base = seq[qpos]

                # Fill all channels for this position
                pile[0, row, col] = self.base_to_intensity(base)
                pile[1, row, col] = min(
                    quals[qpos], 40) / 40.0 if qpos is not None and qpos < len(
                        quals) else 0.5
                pile[2, row, col] = mq
                pile[3, row, col] = 0.0 if is_reverse else 1.0
                pile[4, row, col] = 1.0 if base.upper() == alt.upper() else 0.0
                ref_seq_col = ref_seq[col].upper()
                pile[5, row, col] = 1.0 if base.upper() == ref_seq_col else 0.0
        bam.close()
        fasta.close()
        return pile


class PileupFeaturizer(Featurizer):
    """
    Featurizer that generates pileup images from BAM files and variant
    candidates.

    This featurizer creates multi-channel pileup image representations of
    genomic variants from BAM and FASTA files. These images capture various
    aspects of read alignment around variant positions.

     The pileup images have the following channel structure:
    - Channel 0: Base identity (A=0.25, C=0.5, G=0.75, T=1.0, other=0)
    - Channel 1: Base quality score (normalized to 0-1 range)
    - Channel 2: Mapping quality (normalized to 0-1 range)
    - Channel 3: Read strand (1.0=forward, 0.0=reverse)
    - Channel 4: Match to alternate allele (1.0=match, 0.0=no match)
    - Channel 5: Match to reference allele (1.0=match, 0.0=no match)

    Examples
    --------
    >>> from deepchem.feat import CandidateVariantFeaturizer, PileupFeaturizer
    >>> bamfile_path = 'deepchem/data/tests/example.bam'
    >>> reference_path = 'deepchem/data/tests/sample.fa'
    >>> realign = CandidateVariantFeaturizer()
    >>> datapoint = (bamfile_path, reference_path)
    >>> features = realign.featurize([datapoint])
    >>> candidate_variants = features[0]
    >>> pileup_feat = PileupFeaturizer()
    >>> datapoint = (candidate_variants, reference_path)
    >>> features = pileup_feat.featurize([datapoint])

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with
    Linux or MacOS X. To use Pysam on Windows, use Windows Subsystem for
    Linux(WSL).
    """

    def __init__(self, window=221, height=100, channels=6, labeled=False):
        self.window = window
        self.height = height
        self.channels = channels
        self.labeled = labeled

    def _featurize(self, datapoint):
        """
        datapoint: tuple (bam_path, fasta_path, candidates)
        Returns: DeepChem NumpyDataset (X=images, y=labels or None)
        """
        bam_path, fasta_path, candidates = datapoint
        n = len(candidates)
        X = np.zeros((n, self.channels, self.height, self.window),
                     dtype=np.float32)
        y = np.zeros((n,), dtype=np.int64) if self.labeled else None
        pileup_image = _PileupImage(window=self.window,
                                    height=self.height,
                                    channels=self.channels)

        for i in range(n):
            cand = candidates[i]
            chrom, pos, ref, alt = cand[:4]
            pos = int(pos)
            image = pileup_image.make_image(bam_path, fasta_path, chrom, pos,
                                            ref, alt)
            X[i] = image
            if self.labeled:
                label = int(cand[-1])
                y[i] = label

        return NumpyDataset(X, y)
