"""
Protein Backbone Featurizer for structural diffusion models.
"""
import logging
from typing import Iterable
import numpy as np
from deepchem.feat.base_classes import Featurizer

try:
    from Bio.PDB import PDBParser, is_aa
    has_biopython = True
except ImportError:
    has_biopython = False

logger = logging.getLogger(__name__)


class ProteinBackboneFeaturizer(Featurizer):
    """Featurizes protein structures by extracting backbone atom coordinates.

    This featurizer extracts the N, CA (Cα), and C backbone atom coordinates
    from PDB structures. It is designed for use with protein structure generation
    models like RFDiffusion that operate on backbone geometry.

    Each protein is featurized as an array of shape ``(L, 3, 3)`` where:

    - L is the number of residues
    - The second dimension corresponds to [N, CA, C] atoms
    - The third dimension contains [x, y, z] coordinates in Angstroms

    Since proteins have variable lengths, calling ``featurize()`` returns
    a numpy object array where each element is an ``(L, 3, 3)`` array.

    This class requires BioPython to be installed.

    Parameters
    ----------
    max_length : int, default 512
        Maximum number of residues to extract. Longer proteins will be
        truncated from the center.

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

    def __init__(self, max_length: int = 512):
        """Initialize the ProteinBackboneFeaturizer.

        Parameters
        ----------
        max_length : int, default 512
            Maximum number of residues to extract. Longer proteins will be
            truncated from the center.
        """
        if not has_biopython:
            raise ImportError(
                "ProteinBackboneFeaturizer requires BioPython. "
                "Install it with: pip install biopython")
        self.max_length = max_length

    def featurize(self,
                  datapoints: Iterable[str],
                  log_every_n: int = 1000,
                  **kwargs) -> np.ndarray:
        """Featurize a list of PDB file paths.

        Overrides base ``Featurizer.featurize()`` to return an object-typed
        numpy array, since proteins have variable-length backbones and
        ``np.asarray`` would fail on inhomogeneous shapes.

        Parameters
        ----------
        datapoints : Iterable[str]
            Paths to PDB files to featurize.
        log_every_n : int, default 1000
            Log progress every ``log_every_n`` datapoints.

        Returns
        -------
        np.ndarray
            Object array where each element is an ``(L, 3, 3)``
            float32 array of backbone coordinates.
        """
        datapoints = list(datapoints)
        features = []
        for i, point in enumerate(datapoints):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)
            try:
                features.append(self._featurize(point, **kwargs))
            except Exception:
                logger.warning(
                    "Failed to featurize datapoint %d. "
                    "Appending empty array." % i)
                features.append(np.array([]))

        return np.asarray(features, dtype=object)

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
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', datapoint)

        coords_list = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue

                    try:
                        n_coord = residue['N'].get_coord()
                        ca_coord = residue['CA'].get_coord()
                        c_coord = residue['C'].get_coord()

                        backbone = np.array(
                            [n_coord, ca_coord, c_coord],
                            dtype=np.float32)
                        coords_list.append(backbone)
                    except KeyError:
                        # Skip residues missing backbone atoms
                        continue

            # Only use first model
            break

        if len(coords_list) == 0:
            return np.array([])

        coords = np.stack(coords_list, axis=0)  # (L, 3, 3)

        # Center crop if too long
        L = coords.shape[0]
        if L > self.max_length:
            start = (L - self.max_length) // 2
            coords = coords[start:start + self.max_length]

        return coords
