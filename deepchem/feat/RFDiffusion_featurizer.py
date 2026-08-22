"""Protein Backbone Featurizer for structural diffusion models."""

import logging
from typing import Iterable, List
import numpy as np
import torch
import torch.nn as nn
from deepchem.feat.base_classes import Featurizer
from deepchem.utils.RFDiffusion_utils import get_sinusoid

try:
    from Bio.PDB import PDBParser, is_aa
    has_biopython = True
except ImportError:
    has_biopython = False

logger = logging.getLogger(__name__)


class RFDiffusionFeaturizer(Featurizer):
    """
    RFDiffusionFeaturizer is a featurizer for RFDiffusion models.

    It parses the PDB files and extracts the N, Ca, C atom coordinates, which
    are later used to compute the 1D, 2D and 3D features as inputs to the
    RFDiffusion model.

    Each protien in a file is parsed and the features are returned as a list of:
       1. N, Ca, C backbone atom coordinates
          shape: [batch_size, num_residues, 3, 3]

       2. Embedding of the residue sequence as inputs for the 1D track of
          the RFDiffusion model.

          shape: [batch_size, num_residues, 1, 23]

       3. Rotation and translational matrices of the backbone atoms
          (also known as frames) which are passed as inputs to the
          3D track of the RFDiffusion model.
          As well as used to compute the features for the 2D track of the
          RFDiffusion model.

          shape: [batch_size, R, t]
          R: Rotation matrix of shape [batch_size, num_residue, 3, 3]
          t: Translation matrix of shape [batch_size, num_residue, 3]

        4. The frames computed from the N, Ca, C backbone atom coordinates, are used
           to compute the features for the 2D track of the RFDiffusion model, which
           capture the pairwise distances and orientations between the backbone atoms.

        shape: [batch_size, num_residues, num_residues, d_pair]

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.feat.RFDiffusion_featurizer import RFDiffusionFeaturizer
    >>> features = RFDiffusionFeaturizer()
    >>> datapoint = "101M.pdb"
    >>> _1d_features, _2d_features, frame = features._featurize(datapoint)
    >>> logger.info(_1d_features.shape)
    >>> logger.info(_2d_features.shape)
    >>> logger.info(frame.shape)

    The computed features will further be used as inputs to the RFDiffusion model, to compute the features for the 1D, 2D and 3D tracks.
    shape: [batch_size, [N, d_model], [N, N, d_pair], [R, t]]
    """

    def __init__(self, max_length: int = 512):
        """Initialize the RFDiffusionFeaturizer featurizer.

        Parameters
        ----------
        max_length: int, default 512
            Maximum length of the residue sequence.
        """
        self.max_length = max_length

    def featurize(self,
                  data_points: Iterable[str],
                  log_every_n: int = 1000,
                  **kwargs) -> List[np.ndarray]:
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
            Object array that returns a list containing:
            1. 1D features for the residue sequence
            2. 2D features for the pairwise distances and orientations between the backbone atoms
            3. The frames computed from the N, Ca, C backbone atom coordinates in form of [R, t]

            shape: [batch_size, [N, d_model], [N, N, d_pair], [R, t]]
        """

        datapoints = list(data_points)
        features = []
        for i, d_point in enumerate(datapoints):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)
            try:
                features.append(self._featurize(d_point, **kwargs))
            except Exception:
                logger.warning("Failed to featurize datapoint %d. "
                               "Appending empty array." % i)
                features.append(np.array([]))

        return np.asarray(features, dtype=object)

    def _featurize(self, datapoint: str, **kwargs) -> np.ndarray:
        """
        Featurize a protein file into the RFDiffusion features.

        This calls the following functions to compute the features:

        1. get_backbone_coords: to extract the N, Ca, C backbone atom coordinates from the protein file.
        2. get_frames: to compute the frames for the backbone coordinates.
        3. get_1d_features: to compute the 1D features for the residue sequence.
        4. get_2d_features: to compute the 2D features for the pairwise distances and orientations between the backbone atoms.

        Parameters
        ----------
        datapoint: str
            The path to the protein file.

        Returns
        -------
        _1d_features: np.ndarray
            Array of shape ``N, d_model``  along with sinusoidal embeddings for the residue sequence,
            which represents the 1D features for the residue sequence.
        _2d_features: np.ndarray
            Array of shape ``N, N, d_pair`` which represents the 2D features
            for the pairwise distances and orientations between the backbone atoms.
        frame: np.ndarray
            The frames computed from the N, Ca, C backbone atom coordinates,
            which are used as inputs for 3D track of RFDiffusion model.

        Examples
        --------
        >>> import numpy as np
        >>> from deepchem.feat.RFDiffusion_featurizer import RFDiffusionFeaturizer
        >>> features = RFDiffusionFeaturizer()
        >>> datapoint = "101M.pdb"
        >>> _1d_features, _2d_features, frame = features._featurize(datapoint)
        >>> logger.info(_1d_features.shape)
        >>> logger.info(_2d_features.shape)
        >>> logger.info(frame.shape)
        shape: [batch_size, [N, d_model], [N, N, d_pair], [R, t]]
        """
        backbone_coords = self.get_backbone_coords(datapoint)

        frame = self.get_frames(backbone_coords)

        _1d_features = self.get_1d_features(backbone_coords, frame[1])

        _2d_features = self.get_2d_features(frame[1])

        return _1d_features, _2d_features, frame

    def get_backbone_coords(self, datapoint: str) -> np.ndarray:
        """
        It extracts the N, Ca, C backbone atom coordinates from the protein file.
        These are absolute coordinates which are necessary to compute the frames for the backbone coordinates.

        These will also be used to compute the 1D features for the residue sequence.
        And the 2D features for the pairwise distances and orientations between the backbone atoms.

        Parameters
        ----------
        datapoint: str
            The path to the protein file.

        Returns
        -------
        backbone_coords: np.ndarray
            Array of shape ``N, 3, 3`` which represents the N, Ca, C backbone atom coordinates.
            N: Number of residues in the protein.
            3: Number of atoms per residue (N, Ca, C).
            3: Number of coordinates per atom (x, y, z).

        Examples
        --------
        >>> import numpy as np
        >>> from deepchem.feat.RFDiffusion_featurizer import RFDiffusionFeaturizer
        >>> features = RFDiffusionFeaturizer()
        >>> datapoint = "101M.pdb"
        >>> backbone_coords = features.get_backbone_coords(datapoint)
        >>> logger.info(backbone_coords.shape)
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", datapoint)

        coords = []

        for model in structure:
            for chain in model:
                for residue in chain:

                    if not is_aa(residue.get_resname()):
                        continue

                    try:
                        N_coord = residue['N'].get_coord()
                        Ca_coord = residue['CA'].get_coord()
                        C_coord = residue['C'].get_coord()

                        backbone = np.array([N_coord, Ca_coord, C_coord],
                                            dtype=np.float32)
                        coords.append(backbone)

                    except KeyError:
                        continue

            break  # only one model is supported

        final_coords = np.stack(coords, axis=0)  # (N, 3, 3)

        # Center crop if too long
        N = final_coords.shape[0]
        if N > self.max_length:
            start = (N - self.max_length) // 2
            coords = coords[start:start + self.max_length]

        return coords

    def get_1d_features(self,
                        backbone_coords: np.ndarray,
                        t: int,
                        d_model: int = 64,
                        max_timestep: int = 1000) -> np.ndarray:
        """
        It computes the 1D features for the residue sequence.

        The 1D features represent the sequence of the protein along with their positions in the sequence
        and the timestep. These are used as inputs to the 1D track of the RFDiffusion model.

        This is done by using a learnable mask token embedding and sinusoidal embeddings for the residue sequence and the timestep.

        Parameters
        ----------
        backbone_coords: np.ndarray
            Array of shape ``N, 3, 3`` which represents the N, Ca, C backbone atom coordinates.
        t: int
            The timestep.

        Returns
        -------
        _1d_features: np.ndarray
            Array of shape ``N, d_model``  along with sinusoidal embeddings for the residue sequence,
            which represents the 1D features for the residue sequence.

        """
        N = backbone_coords.shape[0]
        device = backbone_coords.device

        # Base Learnable Mask Token Embedding (Unconditional Sequence representation)
        mask_token_param = nn.Parameter(torch.randn(d_model, device=device))
        seq_features = mask_token_param.view(1, 1,
                                             -1).repeat(1, N,
                                                        1)  # [1, N, d_model]

        # Positional Sinusoids (1 to N)
        pos_indices = torch.arange(N, dtype=torch.float32, device=device)
        raw_pos_sin = get_sinusoid(pos_indices, max_val=N,
                                   channels=32).unsqueeze(0)  # [1, N, 32]

        pos_projector = nn.Linear(32, d_model, device=device)
        pos_features = pos_projector(raw_pos_sin)  # [1, N, d_model]

        # Timestep Sinusoid (t) + Embeddings
        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
        raw_time_sin = get_sinusoid(t_tensor, max_val=max_timestep,
                                    channels=32)  # [1, 32]
        time_projector = nn.Linear(32, d_model, device=device)
        time_features = time_projector(raw_time_sin).view(1, 1, -1).repeat(
            1, N, 1)  # [1, N, d_model]

        track_1d_output = seq_features + pos_features + time_features
        return track_1d_output

    def get_2d_features(self, t_matrix: str) -> np.ndarray:
        """
        It computes the 2D features for the pairwise distances and orientations between the backbone atoms.

        Parameters
        ----------
        t_matrix: np.ndarray
            Array of shape ``N, 3, 3`` which represents the N, Ca, C backbone atom coordinates.

        Returns
        -------
        _2d_features: np.ndarray
            Array of shape ``N, N, 64`` which represents the 2D features
            for the pairwise distances and orientations between the backbone atoms.
        """
        # Calculate continuous pairwise Euclidean distances
        pairwise_distances = torch.cdist(t_matrix, t_matrix, p=2)  # [N, N]

        bins = torch.linspace(2.0, 20.0, 64)
        distance_bins = torch.exp(-(
            (pairwise_distances.unsqueeze(-1) - bins)**2))

        return distance_bins  # [N, N, 64]

    def get_frames(self,
                   backbone: np.ndarray,
                   verbose: bool = False) -> np.ndarray:  # for each backbone
        """
        It computes the frames for the backbone coordinates.

        Parameters
        ----------
        backbone: np.ndarray
            Array of shape ``N, 3, 3`` which represents the N, Ca, C backbone atom coordinates.

        Returns
        -------
        frames: np.ndarray
            Array of shape ``N, 3, 3`` which represents the frames for the backbone coordinates.
            N: Number of residues in the protein.
            3: Number of atoms per residue (N, Ca, C).
            3: Number of coordinates per atom (x, y, z).
        """
        if verbose:
            print("Computing frames for backbone coordinates...")

        N_coords = backbone[:, 0, :]
        CA_coords = backbone[:, 1, :]
        C_coords = backbone[:, 2, :]

        v1 = C_coords - CA_coords  # Gram-Schmidt Orthogonalization
        v2 = N_coords - CA_coords

        # Alignment along the Ca -> C bond
        e1 = v1 / torch.norm(v1, dim=-1, keepdim=True)

        # Axis 2 (y-axis) orthogonal component of the Nitrogen position vector
        u2 = v2 - torch.sum(v2 * e1, dim=-1, keepdim=True) * e1
        e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)

        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        t = CA_coords

        if verbose:
            print("Frames computed successfully.")
            print(f"Rotation matrix shape: {R.shape}")  # [N, 3, 3]
            print(f"Translation matrix shape: {t.shape}")  # [N, 3]

        return [R, t]
