import os
import numpy as np
import logging
from typing import Any, List, Tuple, Optional

from deepchem.feat.base_classes import Featurizer

try:
    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1
except ImportError:
    raise ImportError(
        "biopython library must be installed to run this featurizer.")

logger = logging.getLogger(__name__)

AMINO_ACID_ALPHABET: str = 'ACDEFGHIKLMNPQRSTVWYX'
NUM_AMINO_ACIDS: int = len(AMINO_ACID_ALPHABET)


class ProteinStructureData:
    """
    Container for protein backbone structure and sequence information.

    Each instance represents a single protein (or chain) with backbone atom
    coordinates and the corresponding amino acid sequence.

    Parameters
    ----------
    backbone_coords : np.ndarray
        Backbone atom coordinates where L is the number
        of residues.
        shape: (L, 4, 3)
    sequence : str
        Amino acid sequence string of length L using characters from the
        standard amino acid alphabet.
    chain_mask : np.ndarray, optional
        Binary mask where 1 indicates positions to predict
        and 0 indicates fixed/context positions.
        shape: (L,)
    chain_encoding : np.ndarray, optional
        To indenitfy separate protein chains. Defaults to one chain.
        shape: (L,)
    residue_idx : np.ndarray, optional
        Indexing each residue in the protein sequence, used for referencing the original sequence.
        shape: (L,)
    mask : np.ndarray, optional
        Validity mask where 1 indicates valid residues and
        0 indicates padding or missing coordinates. Defaults to all ones.
        shape: (L,)

    Raises
    ------
    ValueError
        If the length of `backbone_coords` does not match the length of `sequence`.

    Examples
    --------
    >>> import numpy as np
    >>> L = 10
    >>> coords = np.random.randn(L, 4, 3).astype(np.float32)
    >>> seq = 'ACDEFGHIKM'
    >>> structure = ProteinStructureData(backbone_coords=coords, sequence=seq)
    >>> structure.num_residues
    10
    """

    def __init__(
        self,
        backbone_coords: np.ndarray,
        sequence: str,
        chain_mask: Optional[np.ndarray] = None,
        chain_encoding: Optional[np.ndarray] = None,
        residue_idx: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.backbone_coords: np.ndarray[Any, np.dtype[Any]] = np.asarray(
            backbone_coords, dtype=np.float32)
        self.sequence: str = sequence
        L = len(sequence)

        if self.backbone_coords.shape[0] != L:
            raise ValueError(
                'backbone_coords length must match sequence length. '
                f'Got coords with {self.backbone_coords.shape[0]} residues '
                f'and sequence of length {L}.')

        # If chain_mask is not provided, set it to all ones
        if chain_mask is None:
            self.chain_mask: np.ndarray[Any, np.dtype[Any]] = np.ones(
                L, dtype=np.float32)
        else:
            self.chain_mask = np.asarray(chain_mask, dtype=np.float32)

        # If chain_encoding is not provided, set it to all ones
        if chain_encoding is None:
            self.chain_encoding: np.ndarray[Any, np.dtype[Any]] = np.ones(
                L, dtype=np.int32)
        else:
            self.chain_encoding = np.asarray(chain_encoding, dtype=np.int32)

        # If residue_idx is not provided, set it to the range of the sequence
        if residue_idx is None:
            self.residue_idx: np.ndarray[Any, np.dtype[Any]] = np.arange(
                L, dtype=np.int32)
        else:
            self.residue_idx = np.asarray(residue_idx, dtype=np.int32)

        # If mask is not provided, set it to all ones
        if mask is None:
            self.mask: np.ndarray[Any,
                                  np.dtype[Any]] = np.ones(L, dtype=np.float32)
        else:
            self.mask = np.asarray(mask, dtype=np.float32)

    @property
    def num_residues(self) -> int:
        """
        Return the number of residues in the structure.

        Returns
        -------
        int: The length of the amino acid sequence.
        """
        return len(self.sequence)


class _MapperProteinMPNN:
    """
    Helper class to convert `ProteinStructureData` to model-ready tensors.

    This mapper prepares the numpy arrays required by ProteinMPNN
    from a `ProteinStructureData` object, including tokenized sequence indices,
    backbone coordinates, and various masking tensors.

    Parameters
    ----------
    structure : ProteinStructureData
        The raw protein structure data object to be mapped.

    Attributes
    ----------
    structure : ProteinStructureData
        The underlying protein structure data object.
    X : np.ndarray
        Processed backbone coordinates with invalid entries zeroed out.
    mask : np.ndarray
        Processed validity mask filtering out mathematically invalid coordinates.
    S : np.ndarray
        Tokenized integer array representing the amino acid sequence.
    chain_M : np.ndarray
        Processed float array for the chain design mask.
    residue_idx : np.ndarray
        Processed integer array for residue indexing.
    chain_encoding : np.ndarray
        Processed integer array for chain identification.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.randn(5, 4, 3).astype(np.float32)
    >>> structure = ProteinStructureData(backbone_coords=coords, sequence='ACDEF')
    >>> mapper = _MapperProteinMPNN(structure)
    >>> X, S, mask, chain_M, residue_idx, chain_encoding = mapper.values
    >>> X.shape
    (5, 4, 3)
    """

    def __init__(self, structure: ProteinStructureData):
        self.structure = structure
        self._prepare_arrays()

    def _prepare_arrays(self) -> None:
        """
        Convert protein structure data to model-ready numpy arrays.

        Handles NaN filtering, string-to-integer tokenization of the sequence,
        and strict typing of output arrays.

        Example
        -------
        >>> import numpy as np
        >>> from deepchem.feat.ProteinMPNN_featurizer import ProteinStructureData, _MapperProteinMPNN
        >>> coords = np.random.randn(5, 4, 3).astype(np.float32)
        >>> structure = ProteinStructureData(backbone_coords=coords, sequence='ACDEF')
        >>> mapper = _MapperProteinMPNN(structure)
        """
        structure = self.structure

        self.X: np.ndarray[Any,
                           np.dtype[Any]] = structure.backbone_coords.copy()

        self.mask: np.ndarray[Any,
                              np.dtype[Any]] = structure.mask * np.isfinite(
                                  np.sum(self.X, axis=(1, 2))).astype(
                                      np.float32)

        isnan = np.isnan(self.X)
        self.X[isnan] = 0.0

        self.S: np.ndarray[Any, np.dtype[Any]] = np.array([
            AMINO_ACID_ALPHABET.index(a)
            if a in AMINO_ACID_ALPHABET else NUM_AMINO_ACIDS - 1
            for a in structure.sequence
        ],
                                                          dtype=np.int32)

        self.chain_M: np.ndarray[Any,
                                 np.dtype[Any]] = structure.chain_mask.astype(
                                     np.float32)
        self.residue_idx: np.ndarray[
            Any, np.dtype[Any]] = structure.residue_idx.astype(np.int32)
        self.chain_encoding: np.ndarray[
            Any, np.dtype[Any]] = structure.chain_encoding.astype(np.int32)

    @property
    def values(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """
        Return the mapped arrays formatted for model input.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing exactly six formatted numpy arrays:
            - X: backbone coordinates of shape (L, 4, 3)
            - S: sequence token indices of shape (L,)
            - mask: coordinate validity mask of shape (L,)
            - chain_M: designable chain mask of shape (L,)
            - residue_idx: positional indices of shape (L,)
            - chain_encoding: chain identifier encodings of shape (L,)
        """
        return (self.X, self.S, self.mask, self.chain_M, self.residue_idx,
                self.chain_encoding)


class ProteinMPNNFeaturizer(Featurizer):
    """
    This featurizer is for the ProteinMPNN model that parses a raw PDB file, extracts the 3D atomic coordinates
    of the protein backbone (N, CA, C, O), derives the amino acid sequence,
    and formats them into the specific numerical tensors required by the model.
    The model uses a tokenizer to convert the amino acid sequence into a sequence of integer.
    The tokenizer is the same as the one used in the ProteinMPNN model.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> from deepchem.feat.ProteinMPNN_featurizer import ProteinMPNNFeaturizer
    >>> # Create a temporary mock PDB file with 1 residue (Alanine)
    >>> temp_dir = tempfile.mkdtemp()
    >>> mock_pdb_path = os.path.join(temp_dir, 'mock.pdb')
    >>> with open(mock_pdb_path, 'w') as f:
    ...     _ = f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \\n")
    ...     _ = f.write("ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \\n")
    ...     _ = f.write("ATOM      3  C   ALA A   1       2.009   1.424   0.000  1.00  0.00           C  \\n")
    ...     _ = f.write("ATOM      4  O   ALA A   1       1.319   2.441   0.000  1.00  0.00           O  \\n")
    >>> featurizer = ProteinMPNNFeaturizer()
    >>> features = featurizer.featurize([mock_pdb_path])
    >>> len(features)
    1

    References
    ----------
    - ProteinMPNN: Message Passing Neural Network for Protein Sequence Design.
    - https://github.com/aqlaboratory/proteinmpnn
    - https://github.com/RosettaCommons/RFantibody/blob/main/src/rfantibody/proteinmpnn/model/protein_mpnn_utils.py
    """

    def __init__(self, design_chains: Optional[List[str]] = None):
        self.design_chains = design_chains
        self.parser = PDBParser(QUIET=True)
        self.backbone_atoms = ['N', 'CA', 'C', 'O']

    def _featurize(self, datapoint: Any, **kwargs) -> np.ndarray:
        """
        Featurize a single PDB file into ProteinMPNN tensors.

        Parameters
        ----------
        datapoint : str
            File path pointing to the PDB file to be parsed.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            An array containing a single dictionary. The dictionary holds
            the formatted tensor keys: 'X', 'S', 'mask', 'chain_M',
            'residue_idx', and 'chain_encoding'.

        Raises
        ------
        ValueError
            If the `datapoint` is not a string or if the file does not exist.
        """
        if not isinstance(datapoint, str) or not os.path.isfile(datapoint):
            raise ValueError("Datapoint must be a valid file path to a PDB.")

        structure = self.parser.get_structure('protein', datapoint)

        sequence = ""
        coords = []
        chain_mask = []
        chain_encoding = []
        residue_idx = []

        current_chain_idx = 1
        current_residue_idx = 0

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()

                # If design_chains is not provided, set it to all ones
                is_designable = 1.0
                if self.design_chains is not None and chain_id not in self.design_chains:
                    is_designable = 0.0

                for residue in chain:
                    res_name = residue.get_resname()
                    seq_char = protein_letters_3to1.get(res_name.capitalize(),
                                                        'X')
                    sequence += seq_char

                    res_coords = []
                    for atom_name in self.backbone_atoms:
                        if atom_name in residue:
                            res_coords.append(residue[atom_name].get_coord())
                        else:
                            res_coords.append([np.nan, np.nan, np.nan])
                            raise ValueError(
                                f"Atom {atom_name} not found in residue {res_name}"
                            )

                    coords.append(res_coords)
                    chain_mask.append(is_designable)
                    chain_encoding.append(current_chain_idx)
                    residue_idx.append(current_residue_idx)

                    current_residue_idx += 1

                current_chain_idx += 1
            break

        backbone_coords = np.array(coords, dtype=np.float32)

        structure_data = ProteinStructureData(
            backbone_coords=backbone_coords,
            sequence=sequence,
            chain_mask=np.array(chain_mask, dtype=np.float32),
            chain_encoding=np.array(chain_encoding, dtype=np.int32),
            residue_idx=np.array(residue_idx, dtype=np.int32))

        mapper = _MapperProteinMPNN(structure_data)
        X, S, mask, chain_M, res_idx, chain_enc = mapper.values

        features_dict = {
            'X': X,
            'S': S,
            'mask': mask,
            'chain_M': chain_M,
            'residue_idx': res_idx,
            'chain_encoding': chain_enc
        }

        return np.array([features_dict])
