import os
import numpy as np
from typing import List, Optional
from collections import defaultdict

import deepchem as dc
from deepchem.feat import CandidateVariantFeaturizer, PileupFeaturizer
from deepchem.models.torch_models import MobileNetV2Model, TorchModel
from deepchem.data import NumpyDataset

try:
    import pysam
except ImportError:
    pass


class _DeepChemVariant(object):
    """
    Private class with utility functions for variant calling and VCF file operations.

    This class provides static methods for handling Variant Call Format (VCF) file
    operations related to variant calling. It includes functionality for writing
    VCF headers with appropriate metadata, creating empty VCF files, and writing
    variant calls to VCF files based on model predictions.

    The VCF format is a text file format used for storing gene sequence variations.
    It contains meta-information lines, a header line, and data lines each containing
    information about a position in the genome and genotype information on samples
    for each position.
    """

    @staticmethod
    def write_vcf_header(file_obj, fasta_file: str, sample_name: str) -> None:
        """
        Write a complete VCF header to an open file object.

        Creates a standard VCF v4.2 header with appropriate metadata including
        contigs from the reference genome, INFO fields describing variant attributes,
        and FORMAT fields specifying how variant calls are encoded. Contigs are
        extracted directly from the provided FASTA reference file.

        The following INFO fields are included:
        - DP: Total read depth at the position
        - AD: Allelic depths for reference and alternate alleles

        The following FORMAT fields are included:
        - GT: Genotype (0/0=ref, 0/1=het, 1/1=hom)
        - DP: Read depth at this position for this sample
        - AD: Allelic depths for ref and alt alleles
        - GQ: Genotype quality, encoded as a PHRED score

        Parameters
        ----------
        file_obj : file object
            An open file object with write permissions where the header will be written.
        fasta_file : str
            Path to the reference genome FASTA file. Must have an associated .fai index file.
        sample_name : str
            Name of the sample to be included in the header. This name appears in the
            final column of the VCF file.

        Returns
        -------
        None
            The header is written directly to the provided file object.

        """
        fasta = pysam.FastaFile(fasta_file)
        references = fasta.references

        # Write VCF header
        file_obj.write("##fileformat=VCFv4.2\n")
        file_obj.write("##source=DeepVariant-DeepChem\n")

        # Add contigs to header
        for ref in references:
            length = fasta.get_reference_length(ref)
            file_obj.write(f"##contig=<ID={ref},length={length}>\n")

        # Add INFO fields
        file_obj.write(
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total read depth">\n'
        )
        file_obj.write(
            '##INFO=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for ref and alt alleles">\n'
        )

        # Add FORMAT fields
        file_obj.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        file_obj.write(
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        file_obj.write(
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for ref and alt alleles">\n'
        )
        file_obj.write(
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">\n'
        )

        # Write column headers
        file_obj.write(
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n"
        )

        fasta.close()

    @staticmethod
    def write_empty_vcf(output_path: str, fasta_file: str,
                        sample_name: str) -> None:
        """Create an empty VCF file containing only the header.

        This is typically used when no variants are detected in a sample to ensure
        consistent output file creation, regardless of whether variants are found.

        This method creates a properly formatted VCF file with a complete header but
        no variant entries. It opens a file in write mode, writes the header using
        write_vcf_header(), and then closes the file.

        Parameters
        ----------
        output_path : str
            Path where the empty VCF file will be created.
        fasta_file : str
            Path to the reference genome FASTA file. Must have an associated .fai index file.
        sample_name : str
            Name of the sample to be included in the VCF header.

        Returns
        -------
        None
            File is written to disk at the specified output_path.
        """
        with open(output_path, 'w') as f:
            _DeepChemVariant.write_vcf_header(f, fasta_file, sample_name)

    @staticmethod
    def write_vcf(output_path: str,
                  candidates: np.ndarray,
                  genotype_probs: np.ndarray,
                  fasta_file: str,
                  sample_name: str,
                  quality_scale: float = 10.0) -> None:
        """Write variant calls to a VCF file based on model predictions.

        Takes candidate variants and their predicted genotype probabilities,
        converts them to standard VCF format, and writes them to a file.
        Variants are sorted by chromosome and position, and only non-reference
        calls are included in the output.

        Parameters
        ----------
        output_path : str
            Path where the VCF file will be written.
        candidates : np.ndarray
            Array of candidate variants. Each row represents a variant with the format:
            [chrom, pos, ref, alt, alt_count, total_count, ...], where:
            - chrom: Chromosome name (str)
            - pos: 0-based position (int)
            - ref: Reference allele (str)
            - alt: Alternate allele (str)
            - alt_count: Number of reads supporting the alternate allele (int)
            - total_count: Total number of reads at this position (int)
        genotype_probs : np.ndarray
            Model prediction probabilities with shape (n_variants, 3), where each row
            contains probabilities for [reference (0/0), heterozygous (0/1), homozygous (1/1)]
            genotypes for the corresponding variant in candidates.
        fasta_file : str
            Path to reference genome FASTA file. Must have an associated .fai index file.
        sample_name : str
            Name of the sample to be included in the VCF.
        quality_scale : float, default=10.0
            Scaling factor applied to prediction probabilities when converting to
            PHRED-scaled quality scores. Higher values produce higher quality scores.

        Returns
        -------
        None
            File is written to disk at the specified output_path.
        """
        # Get the most likely genotype for each variant
        genotypes = np.argmax(genotype_probs, axis=1)

        # Calculate genotype quality (difference between best and second-best)
        sorted_probs = np.sort(genotype_probs, axis=1)
        gq = np.minimum(99, -10 *
                        np.log10(1.0 - sorted_probs[:, -1] + 1e-10)).astype(int)

        # Group variants by chromosome
        variants_by_chrom = defaultdict(list)

        for i, (candidate, genotype, qual, probs) in enumerate(
                zip(candidates, genotypes, gq, genotype_probs)):
            # Skip reference calls (genotype 0) if desired
            if genotype == 0:
                continue

            # Extract variant fields
            chrom, pos, ref, alt, alt_count, total_count = candidate[:6]
            pos = int(pos) + 1  # Convert to 1-based VCF coordinates

            # Calculate PHRED-scaled quality score
            phred_qual = min(99, int(quality_scale * probs[genotype]))

            # Add to chromosome group with all needed info
            variants_by_chrom[chrom].append((pos, ref, alt, genotype, alt_count,
                                             total_count, phred_qual, qual))

        # Write variants to VCF file
        with open(output_path, 'w') as f:
            _DeepChemVariant.write_vcf_header(f, fasta_file, sample_name)

            # Write variants sorted by chromosome and position
            for chrom in sorted(variants_by_chrom.keys()):
                # Sort variants by position
                sorted_variants = sorted(variants_by_chrom[chrom])

                for (pos, ref, alt, genotype, alt_count, total_count,
                     phred_qual, gq_val) in sorted_variants:
                    ref_count = total_count - alt_count

                    # Format genotype string
                    gt_str = "1/1" if genotype == 2 else "0/1"

                    # Write VCF record
                    f.write(
                        f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{phred_qual}\tPASS\t"
                    )
                    f.write(f"DP={total_count};AD={ref_count},{alt_count}\t")
                    f.write(
                        f"GT:DP:AD:GQ\t{gt_str}:{total_count}:{ref_count},{alt_count}:{gq_val}\n"
                    )


class DeepChemVariant(TorchModel):
    """Deep learning-based variant caller using DeepChem's MobileNetV2Model.

    A genomic variant calling model that implements the DeepVariant approach
    using convolutional neural networks to identify and genotype genomic variants
    from sequencing data. The model follows a three-step process:
    1. Identify candidate variants from sequencing reads in a BAM file
    2. Create pileup image representations of read evidence at each candidate site
    3. Classify each pileup image as reference, heterozygous, or homozygous variant

    The model uses MobileNetV2 as its neural network architecture, making it
    efficient for deployment on various computational platforms while maintaining
    high accuracy. It supports single nucleotide variants (SNVs) and small
    insertions/deletions (indels).

    The model implements a deep learning approach similar to Google's DeepVariant,
    adapted to use DeepChem's MobileNetV2 implementation. The classification task
    has three possible outputs:
    - Class 0: Reference genotype (0/0)
    - Class 1: Heterozygous variant (0/1)
    - Class 2: Homozygous variant (1/1)

    Parameters
    ----------
    model_path : str, optional
        Path to a pre-trained model directory. If provided, the model will be loaded
        from this path. If None, a new model will be initialized.
    window_size : int, default=224
        Width of the pileup image in bases. This value must be divisible by 32 to be
        compatible with MobileNetV2's architecture requirements.
    height : int, default=100
        Height of the pileup image in reads. Represents the maximum number of reads
        to include in each pileup visualization.
    channels : int, default=6
        Number of channels in the pileup image. Each channel represents different
        aspects of the sequencing data (e.g., base identity, quality scores, etc.).
    min_count : int, default=2
        Minimum number of reads supporting an alternate allele to consider it a
        candidate variant.
    min_frac : float, default=0.01
        Minimum fraction of reads that must support an alternate allele to consider
        it a candidate variant (0.01 = 1%).
    realign : bool, default=False
        Whether to perform read realignment during candidate variant detection.
        Enables more accurate identification of indels but increases computation time.
    multiprocessing : bool, default=False
        Whether to use multiprocessing for candidate variant detection to improve
        performance on multi-core systems.
    threads : int, optional
        Number of threads to use if multiprocessing is enabled. If None, will use
        all available CPU cores.

    References
    ----------
    .. [1] Poplin, R., Chang, P. C., Alexander, D., et al. (2018).
       "A universal SNP and small-indel variant caller using deep neural networks."
       Nature biotechnology, 36(10), 983-987.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 window_size: int = 224,
                 height: int = 100,
                 channels: int = 6,
                 min_count: int = 2,
                 min_frac: float = 0.01,
                 realign: bool = False,
                 multiprocessing: bool = False,
                 threads: Optional[int] = None):

        # Ensure window size is compatible with MobileNetV2
        assert window_size % 32 == 0, "window_size must be divisible by 32 for MobileNetV2"

        self.window_size = window_size
        self.height = height
        self.channels = channels
        self.min_count = min_count
        self.min_frac = min_frac
        self.realign = realign
        self.multiprocessing = multiprocessing
        self.threads = threads

        # Initialize featurizers
        self.candidate_featurizer = CandidateVariantFeaturizer(
            min_count=self.min_count,
            min_frac=self.min_frac,
            realign=self.realign,
            multiprocessing=self.multiprocessing,
            threads=self.threads)

        self.pileup_featurizer = PileupFeaturizer(window=self.window_size,
                                                  height=self.height,
                                                  channels=self.channels)

        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.model = MobileNetV2Model(
                n_tasks=1,
                in_channels=self.channels,
                input_size=self.window_size,
                mode='classification',
                n_classes=3  # 0: reference, 1: heterozygous, 2: homozygous
            )
            self.model.restore(model_path)
        else:
            self.model = MobileNetV2Model(n_tasks=1,
                                          in_channels=self.channels,
                                          input_size=self.window_size,
                                          mode='classification',
                                          n_classes=3)

    def call_variants(self,
                      bam_file: str,
                      fasta_file: str,
                      output_vcf: str,
                      batch_size: int = 128,
                      sample_name: Optional[str] = None) -> None:
        """Call genetic variants from a BAM file using the trained model.

        This method implements the complete variant calling pipeline:
        1. Extract candidate variants from the BAM file
        2. Generate pileup images for each candidate
        3. Run inference using the neural network model
        4. Write predicted variants to a VCF file

        Parameters
        ----------
        bam_file : str
            Path to the input BAM file containing aligned sequencing reads.
            The BAM file must be sorted and indexed (.bai file must exist).
        fasta_file : str
            Path to the reference genome FASTA file.
            The FASTA file must be indexed (.fai file must exist).
        output_vcf : str
            Path where the output VCF file with called variants will be written.
        batch_size : int, default=128
            Number of candidate variants to process at once during inference.
            Larger values may be faster but require more memory.
        sample_name : str, optional
            Name of the sample to be included in the VCF. If None, the sample name
            will be extracted from the BAM filename (without extension).

        Returns
        -------
        None
            Results are written to the specified output_vcf file.
        """
        if sample_name is None:
            sample_name = os.path.basename(bam_file).split('.')[0]

        # Extract candidate variants
        datapoint = [(bam_file, fasta_file)]
        candidates = self.candidate_featurizer.featurize(datapoint)[0]

        # Skip if no candidates found
        if len(candidates) == 0:
            _DeepChemVariant.write_empty_vcf(output_vcf, fasta_file,
                                             sample_name)
            return

        # Create pileup images
        dt = [(bam_file, fasta_file, candidates)]
        dataset = self.pileup_featurizer.featurize(dt)[0]

        # Run inference in batches
        predictions = []

        for i in range(0, len(dataset), batch_size):
            batch = dc.data.NumpyDataset(X=dataset.X[i:i + batch_size], y=None)
            batch_preds = self.model.predict(batch)
            predictions.append(batch_preds)

        # Combine predictions
        prediction= np.vstack(predictions)

        # Write to VCF
        _DeepChemVariant.write_vcf(output_vcf, candidates, prediction,
                                   fasta_file, sample_name)

    def train(self,
              bam_files: List[str],
              fasta_files: List[str],
              vcf_files: List[str],
              valid_bam_files: Optional[List[str]] = None,
              valid_fasta_files: Optional[List[str]] = None,
              valid_vcf_files: Optional[List[str]] = None,
              nb_epochs: int = 10,
              batch_size: int = 128,
              learning_rate: float = 0.001,
              model_dir: Optional[str] = None) -> None:
        """Call genetic variants from a BAM file using the trained model.

        This method implements the complete variant calling pipeline:
        1. Extract candidate variants from the BAM file
        2. Generate pileup images for each candidate
        3. Run inference using the neural network model
        4. Write predicted variants to a VCF file

        The method can process large BAM files by analyzing candidates in batches
        to minimize memory usage.

        Parameters
        ----------
        bam_file : str
            Path to the input BAM file containing aligned sequencing reads.
            The BAM file must be sorted and indexed (.bai file must exist).
        fasta_file : str
            Path to the reference genome FASTA file.
            The FASTA file must be indexed (.fai file must exist).
        output_vcf : str
            Path where the output VCF file with called variants will be written.
        batch_size : int, default=128
            Number of candidate variants to process at once during inference.
            Larger values may be faster but require more memory.
        sample_name : str, optional
            Name of the sample to be included in the VCF. If None, the sample name
            will be extracted from the BAM filename (without extension).

        Returns
        -------
        None
            Results are written to the specified output_vcf file.
        """
        # Create training dataset
        train_dataset = self._create_dataset(bam_files, fasta_files, vcf_files)

        # Set up training
        optimizer = dc.models.optimizers.Adam(learning_rate=learning_rate)

        # Train model
        self.model.fit(
            train_dataset,
            nb_epoch=nb_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
        )

        # Save final model
        if model_dir:
            self.model.save_checkpoint(model_dir=model_dir)

    def _create_dataset(self, bam_files: List[str], fasta_files: List[str],
                        vcf_files: List[str]) -> NumpyDataset:
        """
        Internal method that processes raw sequencing data files to create a
        DeepChem NumpyDataset suitable for training. The method:
        1. Extracts candidate variants from each BAM file
        2. Labels candidates using truth variants from VCF files
        3. Creates pileup images for each labeled candidate
        4. Combines data from multiple files into a single dataset

        Parameters
        ----------
        bam_files : List[str]
            List of paths to BAM files containing aligned sequencing reads.
        fasta_files : List[str]
            List of paths to reference FASTA files corresponding to each BAM file.
        vcf_files : List[str]
            List of paths to truth VCF files with known variants for each BAM file.

        Returns
        -------
        NumpyDataset
            A dataset containing:
            - X: Pileup images with shape (n_samples, channels, height, width)
            - y: Labels for each image (0=ref, 1=het, 2=hom)
        """
        if not (len(bam_files) == len(fasta_files) == len(vcf_files)):
            raise ValueError("Number of BAM, FASTA, and VCF files must match")

        all_candidate_datasets = []
        for bam_file, fasta_file, vcf_file in zip(bam_files, fasta_files,
                                                  vcf_files):
            # Create labeled candidate featurizer
            labeled_featurizer = CandidateVariantFeaturizer(
                min_count=self.min_count,
                min_frac=self.min_frac,
                realign=self.realign,
                vcf_path=vcf_file,
                multiprocessing=self.multiprocessing,
                threads=self.threads)

            # Extract labeled candidates
            datapoint = [(bam_file, fasta_file)]
            candidates = labeled_featurizer.featurize(datapoint)[0]

            if len(candidates) > 0:
                # Create labeled pileup images
                labeled_pileup = PileupFeaturizer(window=self.window_size,
                                                  height=self.height,
                                                  channels=self.channels,
                                                  labeled=True)

                dt = [(bam_file, fasta_file, candidates)]
                dataset = labeled_pileup.featurize(dt)[0]
                all_candidate_datasets.append(dataset)

        if not all_candidate_datasets:
            raise ValueError("No labeled variants found in the provided files")

        # Combine datasets
        X = np.vstack([dataset.X for dataset in all_candidate_datasets])
        y = np.hstack([dataset.y for dataset in all_candidate_datasets])

        return NumpyDataset(X, y)
