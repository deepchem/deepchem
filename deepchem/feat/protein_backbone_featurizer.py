"""
Protein Backbone Featurizer for structural diffusion models.
"""
import copy
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from deepchem.feat.base_classes import Featurizer

try:
    from Bio.PDB import PDBParser, is_aa
    has_biopython = True
except ImportError:
    has_biopython = False

logger = logging.getLogger(__name__)


def _empty_backbone_coords() -> np.ndarray:
    """Return an empty backbone tensor with the documented shape."""
    return np.zeros((0, 3, 3), dtype=np.float32)


class ProteinBackboneFeaturizer(Featurizer):
    """Featurizes protein structures by extracting backbone atom coordinates.

    This featurizer extracts the N, CA, and C backbone atom coordinates
    from PDB structures. It is designed for use with protein structure generation
    models like RFDiffusion that operate on backbone geometry.

    Each protein is featurized as an array of shape ``(L, 3, 3)`` where:

    - L is the number of residues
    - The second dimension corresponds to [N, CA, C] atoms
    - The third dimension contains [x, y, z] coordinates in Angstroms

    Since proteins have variable lengths, calling ``featurize()`` returns
    a numpy object array where each element is an ``(L, 3, 3)`` array.
    For multi-model PDB files only the first model is featurized. Standard
    amino-acid residues missing any of N, CA, or C are skipped.

    This class requires BioPython to be installed.

    Parameters
    ----------
    max_length : int or None, default 512
        Maximum number of residues to extract. Longer proteins will be
        center-cropped with a warning, which means the middle
        ``max_length`` residues are kept and residues from both ends are
        dropped. This is useful for RFDiffusion-style models that need a
        fixed upper bound on sequence length while still keeping the
        central part of the fold.

    Examples
    --------
    >>> import deepchem as dc
    >>> import tempfile
    >>> import os
    >>> # Create a minimal PDB file for testing
    >>> pdb_content = '''ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ... ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
    ... ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
    ... END'''
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
    ...     _ = f.write(pdb_content)
    ...     pdb_file = f.name
    >>> featurizer = dc.feat.ProteinBackboneFeaturizer()
    >>> features = featurizer.featurize([pdb_file])
    >>> features[0].shape
    (1, 3, 3)
    >>> os.unlink(pdb_file)

    References
    ----------
    .. [1] Watson, J. L., et al. "De novo design of protein structure and
       function with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
    """

    def __init__(self, max_length: Optional[int] = 512):
        """Initialize the ProteinBackboneFeaturizer.

        Parameters
        ----------
        max_length : int or None, default 512
            Maximum number of residues to extract. Longer proteins will be
            center-cropped with a warning, which means the middle
            ``max_length`` residues are kept and residues from both ends are
            dropped. This keeps a consistent maximum size for diffusion
            training without always biasing the crop toward the N-terminus
            or C-terminus. If None, no length limit is applied.
        """
        if not has_biopython:
            raise ImportError("ProteinBackboneFeaturizer requires BioPython. "
                              "Install it with: pip install biopython")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive or None")
        self.max_length = max_length
        self._last_metadata: Dict[str, Dict[str, Any]] = {}

    def get_metadata(self, datapoint: str) -> Dict[str, Any]:
        """Return metadata recorded during the most recent featurization.

        Parameters
        ----------
        datapoint : str
            Path to a PDB file that was previously featurized.

        Returns
        -------
        dict
            Metadata including original length, skipped residues, chains,
            model id, and truncation details. Returns an empty dict if the
            datapoint has not been featurized by this instance.

        Examples
        --------
        >>> import deepchem as dc
        >>> import tempfile
        >>> import os
        >>> pdb_content = '''ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
        ... ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
        ... ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
        ... END'''
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        ...     _ = f.write(pdb_content)
        ...     pdb_file = f.name
        >>> featurizer = dc.feat.ProteinBackboneFeaturizer()
        >>> _ = featurizer.featurize([pdb_file])
        >>> meta = featurizer.get_metadata(pdb_file)
        >>> meta['returned_length']
        1
        >>> os.unlink(pdb_file)
        """
        return copy.deepcopy(self._last_metadata.get(datapoint, {}))

    def _featurize(self, datapoint: str, **kwargs) -> np.ndarray:
        """Extract backbone coordinates from a PDB file.

        Parameters
        ----------
        datapoint : str
            Path to a PDB file.

        Returns
        -------
        np.ndarray
            Array of shape ``(L, 3, 3)`` containing backbone atom
            coordinates, where L is the number of residues, or an
            empty array on failure.
        """
        try:
            return self._parse_backbone(datapoint)
        except Exception as exc:
            self._last_metadata.pop(datapoint, None)
            logger.warning(
                "Failed to featurize datapoint %s: %s. Appending empty array.",
                datapoint, exc)
            return _empty_backbone_coords()

    def _parse_backbone(self, datapoint: str) -> np.ndarray:
        """Parse backbone coordinates from a PDB file.

        Parameters
        ----------
        datapoint : str
            Path to a PDB file.

        Returns
        -------
        np.ndarray
            Array of shape ``(L, 3, 3)`` of backbone atom coordinates.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', datapoint)

        coords_list: List[np.ndarray] = []
        metadata: Dict[str, Any] = {
            'model_id': None,
            'chain_ids': [],
            'original_length': 0,
            'returned_length': 0,
            'skipped_residues': 0,
            'truncated': False,
            'crop_start': None,
        }
        for model in structure:
            metadata['model_id'] = model.id
            for chain in model:
                metadata['chain_ids'].append(chain.id)
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue

                    try:
                        n_coord = residue['N'].get_coord()
                        ca_coord = residue['CA'].get_coord()
                        c_coord = residue['C'].get_coord()

                        backbone = np.array([n_coord, ca_coord, c_coord],
                                            dtype=np.float32)
                        coords_list.append(backbone)
                    except KeyError:
                        metadata['skipped_residues'] += 1
                        continue

            # Only use first model
            break

        if len(coords_list) == 0:
            self._last_metadata[datapoint] = metadata
            return _empty_backbone_coords()

        coords = np.stack(coords_list, axis=0)  # (L, 3, 3)
        metadata['original_length'] = int(coords.shape[0])

        # Center crop if too long
        L = coords.shape[0]
        if self.max_length is not None and L > self.max_length:
            start = (L - self.max_length) // 2
            coords = coords[start:start + self.max_length]
            metadata['truncated'] = True
            metadata['crop_start'] = int(start)
            logger.warning(
                "Protein %s has %d residues; center-cropped to max_length=%d",
                datapoint, L, self.max_length)

        metadata['returned_length'] = int(coords.shape[0])
        self._last_metadata[datapoint] = metadata
        return coords
