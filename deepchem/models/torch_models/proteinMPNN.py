"""
ProteinMPNN: Message Passing Neural Network for Protein Sequence Design.

This module implements the ProteinMPNN architecture described in Dauparas et al.
(Science, 2022) for designing amino acid sequences from protein backbone structures.
"""

import numpy as np

from typing import Optional, Tuple

# Standard amino acid alphabet used by ProteinMPNN (20 AAs + unknown 'X').
AMINO_ACID_ALPHABET: str = 'ACDEFGHIKLMNPQRSTVWYX'
NUM_AMINO_ACIDS: int = len(AMINO_ACID_ALPHABET)


class ProteinStructureData:
    """Container for protein backbone structure and sequence information.

    Each instance represents a single protein (or chain) with backbone atom
    coordinates and the corresponding amino acid sequence. This class is the
    primary input format for `ProteinMPNNModel`.

    Parameters
    ----------
    backbone_coords: np.ndarray
        Backbone atom coordinates of shape ``(L, 4, 3)`` where L is the number
        of residues. The four atoms per residue are ordered as N, CA, C, O.
    sequence: str
        Amino acid sequence string of length L using characters from
        :data:`AMINO_ACID_ALPHABET`.
    chain_mask: np.ndarray, optional (default None)
        Binary mask of shape ``(L,)`` where 1 indicates positions to design
        (predict) and 0 indicates fixed/context positions. Defaults to all ones.
    chain_encoding: np.ndarray, optional (default None)
        Integer chain identifiers of shape ``(L,)``. Defaults to ones.
    residue_idx: np.ndarray, optional (default None)
        Residue index offsets of shape ``(L,)``. Defaults to ``0, 1, ..., L-1``.
    mask: np.ndarray, optional (default None)
        Validity mask of shape ``(L,)`` where 1 indicates valid residues and
        0 indicates padding or missing coordinates. Defaults to all ones.

    Example
    -------
    >>> import numpy as np
    >>> from deepchem.models.torch_models.proteinMPNN import ProteinStructureData
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
        self.backbone_coords = np.asarray(backbone_coords, dtype=np.float32)
        self.sequence = sequence
        L = len(sequence)

        if self.backbone_coords.shape[0] != L:
            raise ValueError(
                'backbone_coords length must match sequence length. '
                f'Got coords with {self.backbone_coords.shape[0]} residues '
                f'and sequence of length {L}.')

        if chain_mask is None:
            self.chain_mask = np.ones(L, dtype=np.float32)
        else:
            self.chain_mask = np.asarray(chain_mask, dtype=np.float32)

        if chain_encoding is None:
            self.chain_encoding = np.ones(L, dtype=np.int32)
        else:
            self.chain_encoding = np.asarray(chain_encoding, dtype=np.int32)

        if residue_idx is None:
            self.residue_idx = np.arange(L, dtype=np.int32)
        else:
            self.residue_idx = np.asarray(residue_idx, dtype=np.int32)

        if mask is None:
            self.mask = np.ones(L, dtype=np.float32)
        else:
            self.mask = np.asarray(mask, dtype=np.float32)

    @property
    def num_residues(self) -> int:
        """Number of residues in the structure."""
        return len(self.sequence)


class _MapperProteinMPNN:
    """Helper class to convert :class:`ProteinStructureData` to model tensors.

    This mapper prepares the numpy arrays required by :class:`ProteinMPNN`
    from a :class:`ProteinStructureData` object, including sequence indices,
    backbone coordinates, and masking tensors.

    Parameters
    ----------
    structure: ProteinStructureData
        Protein structure data object.

    Example
    -------
    >>> import numpy as np
    >>> from deepchem.models.torch_models.proteinMPNN import (
    ...     ProteinStructureData, _MapperProteinMPNN)
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

    def _prepare_arrays(self):
        """
        Convert structure fields to model-ready numpy arrays.
        Example
        -------
        >>> import numpy as np
        >>> from deepchem.models.torch_models.proteinMPNN import ProteinStructureData, _MapperProteinMPNN
        >>> coords = np.random.randn(5, 4, 3).astype(np.float32)
        >>> structure = ProteinStructureData(backbone_coords=coords, sequence='ACDEF')
        >>> mapper = _MapperProteinMPNN(structure)
        >>> X, S, mask, chain_M, residue_idx, chain_encoding = mapper.values
        """
        structure = self.structure

        self.X: np.ndarray = structure.backbone_coords.copy()

        self.mask: np.ndarray = structure.mask * np.isfinite(
            np.sum(self.X, axis=(1, 2))).astype(np.float32)

        isnan = np.isnan(self.X)
        self.X[isnan] = 0.0

        self.S: np.ndarray = np.array([
            AMINO_ACID_ALPHABET.index(a)
            if a in AMINO_ACID_ALPHABET else NUM_AMINO_ACIDS - 1
            for a in structure.sequence
        ],
                                      dtype=np.int32)

        self.chain_M: np.ndarray = structure.chain_mask.astype(np.float32)
        self.residue_idx: np.ndarray = structure.residue_idx.astype(np.int32)
        self.chain_encoding: np.ndarray = structure.chain_encoding.astype(
            np.int32)

    @property
    def values(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """Return mapped arrays for model input.

        Returns
        -------
        Tuple containing:
            - backbone_coords (L, 4, 3)
            - sequence_indices (L,)
            - validity_mask (L,)
            - design_mask (L,)
            - residue_idx (L,)
            - chain_encoding (L,)
        """
        return (self.X, self.S, self.mask, self.chain_M, self.residue_idx,
                self.chain_encoding)
