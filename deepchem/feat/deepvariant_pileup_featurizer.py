import numpy as np
from deepchem.feat import Featurizer
from deepchem.data import ImageDataset
from typing import List


class PileupFeaturizer(Featurizer):
    """
    Generates Pileup Images from haplotype windows.

    PileupFeaturizer generates pileup images from DNA sequence
    data in a specific genomic window. Each image encodes sequence
    information across multiple channels, representing features
    like base identity, base quality, mapping quality, strand
    orientation, variant support, and differences from the
    reference sequence. This featurizer decodes one-hot encoded
    reference sequences, aligns haplotypes, and calculates intensity
    values for each feature in the pileup image. The generated images
    are used for downstream tasks, such as variant calling.

    Examples
    --------
    >>> from deepchem.feat import RealignerFeaturizer, PileupFeaturizer
    >>> bamfile_path = 'deepchem/data/tests/example.bam'
    >>> reference_path = 'deepchem/data/tests/sample.fa'
    >>> realigner_feat = RealignerFeaturizer()
    >>> windows_haplotypes = realigner_feat.featurize((bamfile_path, reference_path))
    >>> pileup_feat = PileupFeaturizer()
    >>> features = pileup_feat.featurize((windows_haplotypes, reference_path))
    >>> features
    <ImageDataset X.shape: (15, 100, 221, 6), y.shape: (15,), w.shape: (15,), ids: [0 1 2 ... 12 13 14], task_names: [0]>


    """

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

    def _featurize(self, datapoint):
        """
        Featurizes a datapoint by generating pileup images.

        Args:
            datapoint (Tuple[List[Any], str]): A tuple containing
            haplotypes and reference file path.

        Returns:
            ImageDataset: An ImageDataset containing the images and labels.

        """
        windows_haplotypes = datapoint[0]
        reference_file_path = datapoint[1]

        from deepchem.data import FASTALoader

        fasta_loader = FASTALoader(None, False, False)
        fasta_dataset = fasta_loader.create_dataset(reference_file_path)

        one_hot_encoded_sequences = fasta_dataset.X
        decoded_sequences = []

        # Convert the one-hot encoded sequences to strings
        for seq in one_hot_encoded_sequences:
            decoded_seq = self.decode_one_hot(seq)
            decoded_sequences.append(decoded_seq)

        # Map the sequences to chrom names
        chrom_names = ["chr1", "chr2"]

        reference_seq_dict = {
            chrom_names[i]: seq for i, seq in enumerate(decoded_sequences)
        }

        def base_to_intensity(base):
            if base == "A":
                return 0.25
            elif base == "C":
                return 0.5
            elif base == "G":
                return 0.75
            elif base == "T":
                return 1.0
            else:
                return 0

        def get_base_quality_intensity(quality):
            return quality / 40.0

        def get_mapping_quality_intensity(mapping_quality):
            return mapping_quality / 60.0

        def get_strand_intensity(is_reverse):
            return 1.0 if is_reverse else 0.0

        def get_supports_variant_intensity(read, haplotype):
            read_bases = read[1]
            query_length = read[2]
            ref_bases = haplotype[:query_length]
            mismatch = False
            for rb, hb in zip(read_bases, ref_bases):
                if rb != hb:
                    mismatch = True
                    break
            return 1.0 if mismatch else 0.5

        def get_diff_from_ref_intensity(base, ref_base):
            return 1.0 if base != ref_base else 0.25

        height = 221
        width = 100
        num_channels = 6

        images = []
        labels = []

        for window in windows_haplotypes:
            chrom, start, end = window['span']
            haplotypes = window['haplotypes']
            realigned_reads = window['realigned_reads']

            if not haplotypes:
                continue

            variant_pos = (start + end) // 2  # Variant position in the center
            start = max(0, variant_pos - height // 2)
            end = start + height

            ref_sequence = reference_seq_dict[chrom][start:end + 1]
            image = np.zeros((width, height, num_channels))

            for i, read in enumerate(realigned_reads):
                if i >= width:
                    break
                for j in range(read[2]):
                    ref_pos = read[4] + j - start
                    if ref_pos < 0 or ref_pos >= height or ref_pos >= len(
                            ref_sequence):
                        continue

                    base = read[1][j]
                    base_quality = read[8][j]
                    mapping_quality = read[6]
                    is_reverse = read[7]
                    supports_variant = get_supports_variant_intensity(
                        read, haplotypes[0])
                    diff_from_ref = get_diff_from_ref_intensity(
                        base, ref_sequence[ref_pos])

                    image[i, ref_pos, 0] = base_to_intensity(base)
                    image[i, ref_pos,
                          1] = get_base_quality_intensity(base_quality)
                    image[i, ref_pos,
                          2] = get_mapping_quality_intensity(mapping_quality)
                    image[i, ref_pos, 3] = get_strand_intensity(is_reverse)
                    image[i, ref_pos, 4] = supports_variant
                    image[i, ref_pos, 5] = diff_from_ref

            images.append(image)  # Store the generated image
            labels.append(1)

        # Convert the image list to numpy array
        images_array = np.array(images)
        labels_array = np.array(labels)

        # Create ImageDataset
        image_dataset = ImageDataset(X=images_array, y=labels_array)

        return image_dataset
