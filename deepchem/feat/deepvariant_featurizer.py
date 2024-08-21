import numpy as np
from collections import defaultdict
from deepchem.feat import Featurizer
from typing import List, Dict, Tuple, Any, Optional

try:
    import dgl
    import torch
except ImportError:
    pass


class _Realigner(object):
    """
    A private class for realigning sequencing reads to a reference sequence.
    This class provides methods for left-aligning indels, processing pileup
    information, generating pileups and reads, selecting candidate regions,
    fetching reads, building De Bruijn graphs, pruning graphs, and generating
    candidate haplotypes.

    """

    def left_align_indel(self, seq: str, pos: int,
                         indel: str) -> Tuple[int, str]:
        """
        Left align an indel by shifting it to the left as
        much as possible. This function shifts an indel to the left
        if the preceding bases match the end of the indel sequence,
        making the indel left-aligned.

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

    def process_pileup(
            self, pileup_info: List[Dict[str,
                                         Any]], reference_seq_dict: Dict[str,
                                                                         str],
            allele_counts: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
        """
        Process pileup information to extract allele counts. This function
        processes each pileup column to count the occurrences of each
        allele at each position, updating the allele counts dictionary.

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
        self, bamfiles, reference_seq_dict
    ) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Any]]:
        """
        Generate pileup and reads from BAM and reference FASTA files. This
        function generates pileup information and reads from the provided
        BAM files and reference sequences, returning both allele counts and reads.

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
        allele_counts: Dict[Tuple[str, int], Dict[str, Any]] = {}
        reads = []

        for x in bamfiles:
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
        Update counts in a window. This function increments the count
        for each position in the specified range by the given count.

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
        Select candidate regions based on allele counts. This function
        identifies candidate regions with significant allele counts
        and groups adjacent positions into regions.

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

    def fetchreads(self, bamfiles: List[Tuple[str, Any, int, str, int, Any, int,
                                              Any]], chrom: str, start: int,
                   end: int) -> List[str]:
        """
        Fetch reads from BAM files for a specific chromosome and region.
        This function extracts reads from BAM files that overlap with the
        specified chromosome and region.

        Parameters
        ----------

        bamfiles : List[Tuple[str, Any, int, str, int, Any, int, Any]]
            List of BAM file records, where each record is a tuple containing
            information about reads in the BAM file.
        chrom : str
            Chromosome name to fetch reads from.
        start : int
            Start position of the region.
        end : int
            End position of the region.

        Returns
        -------

        List[str]
            List of read sequences that fall within the specified region.
        """
        reads: List[str] = []
        for bamfile in bamfiles:
            refname = bamfile[3]
            refstart = bamfile[4]
            refend = refstart + bamfile[2]

            if refname == chrom and refstart < end and refend > start:
                reads.append(bamfile[1])
        return reads

    def build_debruijn_graph(
        self, ref: str, reads: List[str], k: int
    ) -> Tuple[Optional[Any], Optional[Dict[str, int]], Optional[Dict[int,
                                                                      str]]]:
        """
        Build a De Bruijn graph from reference and reads. This function
        constructs a De Bruijn graph from k-mers found in the reference
        and reads, where nodes represent k-mers and edges represent
        overlaps between k-mers.

        Parameters
        ----------

        ref : str
            Reference sequence as a string.
        reads : List[str]
            List of read sequences.
        k : int
            Length of k-mers.

        Returns
        -------

        Tuple[Optional[Any], Optional[Dict[str, int]], Optional[Dict[int, str]]]
            A tuple containing:
            - De Bruijn graph (dgl.DGLGraph)
            - Dictionary mapping k-mer strings to node IDs (Dict[str, int])
            - Dictionary mapping node IDs to k-mer strings (Dict[int, str])
        """
        kmer_counts: Dict[str, int] = defaultdict(int)

        def get_kmers(sequence: str, k: int):
            for i in range(len(sequence) - k + 1):
                yield sequence[i:i + k]

        # Count k-mers in reference
        for kmer in get_kmers(ref, k):
            kmer_counts[kmer] += 1

        # Count k-mers in reads
        for read in reads:
            for kmer in get_kmers(read, k):
                kmer_counts[kmer] += 1

        kmer_to_id: Dict[str, int] = {}
        id_to_kmer: Dict[int, str] = {}
        node_id: int = 0
        edges: List[Tuple[int, int]] = []
        edge_weights: List[int] = []

        for kmer in kmer_counts:
            if kmer not in kmer_to_id:
                kmer_to_id[kmer] = node_id
                id_to_kmer[node_id] = kmer
                node_id += 1

        # Add edges between overlapping k-mers
        for kmer in kmer_counts:
            suffix = kmer[1:]
            for next_kmer in kmer_counts:
                if suffix == next_kmer[:-1]:
                    edges.append((kmer_to_id[kmer], kmer_to_id[next_kmer]))
                    edge_weights.append(kmer_counts[next_kmer])

        if not edges:
            return None, None, None

        src, dst = zip(*edges)
        G = dgl.graph((src, dst), num_nodes=node_id)
        G.ndata['weight'] = torch.tensor([kmer_counts[k] for k in kmer_counts],
                                         dtype=torch.float32)
        G.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)

        return G, kmer_to_id, id_to_kmer

    def prune_debruijn_graph(self, G: Any, min_edge_weight: float) -> Any:
        """
        This function removes edges with weights below the specified threshold
        and removes nodes that become isolated after pruning.

        Parameters
        ----------

        G : Any
            The original De Bruijn graph.
        min_edge_weight : float
            The minimum edge weight threshold for keeping an edge.

        Returns
        -------

        Any
            The pruned subgraph.
        """
        edge_weights = G.edata['weight']
        mask = edge_weights >= min_edge_weight
        edges_to_keep = mask.nonzero(as_tuple=False).squeeze().tolist()

        # Create a subgraph with the edges to keep
        subgraph = dgl.edge_subgraph(G, edges_to_keep, relabel_nodes=False)

        # Identify and remove isolated nodes
        isolated_nodes = (subgraph.in_degrees() == 0) & (subgraph.out_degrees()
                                                         == 0)
        isolated_nodes = isolated_nodes.nonzero(
            as_tuple=False).squeeze().tolist()
        subgraph = dgl.remove_nodes(subgraph, isolated_nodes)

        return subgraph

    def candidate_haplotypes(self, G: Any, k: int,
                             id_to_kmer: Optional[Dict[int, str]]) -> List[str]:
        """
        Generate candidate haplotypes from the De Bruijn graph. This function
        traverses the De Bruijn graph to generate potential haplotypes by
        combining k-mers along paths from start to end nodes.

        Parameters
        ----------

        G : dgl.DGLGraph
            The De Bruijn graph.
        k : int
            The k-mer length.
        id_to_kmer : Dict[int, str]
            A dictionary mapping node IDs to k-mers.

        Returns
        -------

        List[str]
            Sorted list of candidate haplotypes.
        """
        haplotypes: List[str] = []
        nodes = list(G.nodes().numpy())

        if not nodes:
            return haplotypes

        if id_to_kmer is None:
            return haplotypes

        start_node = nodes[0]
        end_node = nodes[-1]

        def dfs(node: int, path: List[int]) -> None:
            path.append(node)
            if node == end_node:
                haplotype = id_to_kmer[path[0]]
                for p in path[1:]:
                    haplotype += id_to_kmer[p][-1]
                haplotypes.append(haplotype)
            else:
                for succ in G.successors(node).numpy():
                    dfs(succ, path[:])

        dfs(start_node, [])

        return sorted(haplotypes)

    def process_candidate_windows(
        self, candidate_regions: List[Tuple[str, int, int, int]],
        bamfiles: List[Any], reference_seq_dict: Dict[str, str]
    ) -> List[Tuple[str, int, int, int, int, List[str]]]:
        """
        Process candidate regions to generate candidate windows with haplotypes.

        Parameters
        ----------

        candidate_regions : List[Tuple[str, int, int, int]]
            List of candidate regions.
        bamfiles : List[Any]
            List of BAM file data.
        reference_seq_dict : Dict[str, str]
            Dictionary with chromosome names as keys and reference sequences as values.

        Returns
        -------

        List[Tuple[str, int, int, int, int, List[str]]]
            List of candidate windows with haplotypes.
        """
        candidate_windows: List[Tuple[str, int, int, int, int, List[str]]] = []

        for chrom, start, end, count in candidate_regions:
            window_reads = self.fetchreads(bamfiles, chrom, start, end)
            ref_sequence = reference_seq_dict[chrom][start:end + 1]

            kmin, kmax, step_k = 15, 21, 2
            found_graph = False

            for k in range(kmin, kmax + 1, step_k):
                dbg, kmer_to_id, id_to_kmer = self.build_debruijn_graph(
                    ref_sequence, window_reads, k)

                if dbg is None:
                    continue

                if kmer_to_id is None:
                    continue

                dbg = self.prune_debruijn_graph(dbg, min_edge_weight=2)

                if dbg.number_of_nodes() == 0:
                    continue

                found_graph = True
                candidate_haplotypes_list = self.candidate_haplotypes(
                    dbg, k, id_to_kmer)

                if candidate_haplotypes_list and candidate_haplotypes_list != [
                        ref_sequence
                ]:
                    candidate_windows.append((chrom, start, end, count, k,
                                              candidate_haplotypes_list))
                    break

            if not found_graph:
                candidate_windows.append(
                    (chrom, start, end, count, k, [ref_sequence]))

        return candidate_windows


class RealignerFeaturizer(Featurizer):
    """
    Realigns reads and generates candidate regions for variant calling.

    More features can be added to this class in the future.

    Note
    ----
    This class requires Pysam, DGL and Pytorch to be installed. Pysam can be
    used with Linux or MacOS X. To use Pysam on Windows, use Windows
    Subsystem for Linux(WSL).

    """

    def __init__(self):
        self.realigner = _Realigner()

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
        Featurizes a datapoint by generating candidate regions and reads.

        Args:
            datapoint (Tuple[str, str]): A tuple containing two strings
            representing allele counts and reads.

        Returns:
            Tuple[List[Tuple[str, int, int, int]], List[Any]]: A tuple
            containing the candidate regions and reads.

        """
        bamfile_path = datapoint[0]
        reference_file_path = datapoint[1]

        from deepchem.data import BAMLoader, FASTALoader

        bam_loader = BAMLoader(get_pileup=True)
        bam_dataset = bam_loader.create_dataset(bamfile_path)
        bamfiles = bam_dataset.X

        fasta_loader = FASTALoader(None, False, False)
        fasta_dataset = fasta_loader.create_dataset(reference_file_path)

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

        allele_counts, reads = self.realigner.generate_pileup_and_reads(
            bamfiles, reference_seq_dict)
        candidate_regions = self.realigner.select_candidate_regions(
            allele_counts)
        candidate_windows = self.realigner.process_candidate_windows(
            candidate_regions, bamfiles, reference_seq_dict)
        return candidate_windows
