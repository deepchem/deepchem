"""RFDiffusion All-Atom: ligand-conditioned protein backbone generation.

This is the All-Atom integration PR. It composes the pieces built in the
earlier All-Atom PRs into a working model:

- the multi-track backbone denoiser and combined IGSO(3) + translational
  diffusion from ``rfdiffusion_multitrack.py``,
- ligand parsing and the ligand clash loss from ``rfdiffusion_ligand.py``
  / ``rfdiffusion_losses.py``,
- cross-attention conditioning from ``rfdiffusion_conditioning.py``.

Scope
-----
The real RFdiffusion All-Atom model (RoseTTAFold All-Atom) [Krishna2024]_
jointly diffuses full side-chain atom coordinates for the protein *and*
the ligand, using a per-residue-type atom14 representation. DeepChem's
RFdiffusion stack only has a backbone (N, CA, C) protein representation
so far -- there is no side-chain atom14 featurizer to build a matching
all-atom protein diffusion process on top of. Rather than invent one, or
silently claim parity that a backbone-only model can't have, this PR
implements the piece that *is* faithfully reproducible with what
DeepChem currently has: ligand-conditioned backbone generation. The
protein backbone is diffused exactly as in ``RFDiffusionModel``
(rigid frames, IGSO(3) rotational diffusion); a fixed ligand's atoms
(all-atom resolution, from ``parse_ligand_file``) are embedded and
cross-attended into every denoising step via ``BinderCrossAttention``,
and a protein-ligand clash penalty (the same steric-overlap idea as
``ligand_clash_loss``, applied here between protein and ligand atoms
rather than within a single ligand) discourages generated backbone atoms
from overlapping the ligand during training. This is the same "generate
a chain that respects a fixed all-atom context" mechanism the reference
implementation uses for binder and small-molecule-pocket design, without
requiring a full joint side-chain/ligand diffusion process neither this
codebase nor its test data can currently validate.

References
----------
.. [Krishna2024] Krishna, R., et al. "Generalized biomolecular modeling
   and design with RoseTTAFold All-Atom." Science 384 (2024) eadl2528.
.. [Watson2023] Watson, J. L., et al. "De novo design of protein
   structure and function with RFdiffusion." Nature 620 (2023)
   1089-1100.

Notes
-----
This module requires PyTorch to be installed.
"""

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('RFDiffusionAA requires PyTorch to be installed.')

from deepchem.data import Dataset
from deepchem.models.torch_models.layers import (
    CosineSchedule,
    PositionalEncoding,
    ResidueEmbedding,
    SinusoidalTimestepEmbedding,
)
from deepchem.models.torch_models.rfdiffusion_conditioning import (
    BinderCrossAttention,)
from deepchem.models.torch_models.rfdiffusion_frames import (
    build_backbone_frames,)
from deepchem.models.torch_models.rfdiffusion_multitrack import (
    backbone_coords_from_frames,
    multitrack_frame_loss,
    sample_noisy_frames,
    so3_x0_reverse_step,
    translation_posterior_step,
)
from deepchem.models.torch_models.rfdiffusion_sequence_track import (
    RFDiffusionMultiTrackStack,)
from deepchem.models.torch_models.rfdiffusion_so3 import IGSO3, log_beta_schedule
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.utils.rfdiffusion_ligand import (LIGAND_ATOM_TYPES,
                                               LigandPointCloud)

logger = logging.getLogger(__name__)


class LigandContextEmbedding(nn.Module):
    """Embeds a fixed ligand point cloud into per-atom conditioning features.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension, matching the protein denoiser's
        single-track channel size.
    num_atom_types : int, default len(LIGAND_ATOM_TYPES)
        Size of the atom-type vocabulary (see
        :data:`~deepchem.utils.rfdiffusion_ligand.LIGAND_ATOM_TYPES`).

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_aa import (
    ...     LigandContextEmbedding)
    >>> embed = LigandContextEmbedding(embed_dim=16)
    >>> atom_types = torch.zeros(2, 5, dtype=torch.long)
    >>> coords = torch.randn(2, 5, 3)
    >>> out = embed(atom_types, coords)
    >>> out.shape
    torch.Size([2, 5, 16])
    """

    def __init__(
        self, embed_dim: int,
        num_atom_types: int = len(LIGAND_ATOM_TYPES)) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        self.atom_type_embedding = nn.Embedding(num_atom_types, embed_dim)
        self.coord_proj = nn.Linear(3, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, atom_types: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        """Embed ligand atoms.

        Parameters
        ----------
        atom_types : torch.Tensor
            Integer atom-type indices, shape ``(batch, num_atoms)``.
        coords : torch.Tensor
            Ligand atom coordinates, shape ``(batch, num_atoms, 3)``, in
            the same normalized coordinate frame as the protein.

        Returns
        -------
        torch.Tensor
            Per-atom embeddings, shape ``(batch, num_atoms, embed_dim)``.
        """
        return self.norm(
            self.atom_type_embedding(atom_types) + self.coord_proj(coords))


class RFDiffusionAADenoiser(nn.Module):
    """Frame-based denoiser with ligand cross-attention conditioning.

    Identical to
    :class:`~deepchem.models.torch_models.rfdiffusion_multitrack.RFDiffusionMultiTrackDenoiser`,
    except the protein's single representation is updated by attending
    onto a fixed ligand's embedded atoms
    (:class:`LigandContextEmbedding` + :class:`~deepchem.models.torch_models.rfdiffusion_conditioning.BinderCrossAttention`)
    before entering the multi-track stack, so every block's IPA and
    pair-track updates are already ligand-aware.

    Parameters
    ----------
    embed_dim : int, default 128
        Single-track channel size.
    pair_dim : int, default 64
        Pair-track channel size.
    time_dim : int, default 128
        Dimension of the sinusoidal timestep embedding.
    num_blocks : int, default 2
        Number of stacked multi-track blocks.
    num_heads : int, default 8
        Attention heads for the single track and IPA.
    pair_num_heads : int, default 4
        Attention heads for triangular self-attention.
    ligand_num_heads : int, default 8
        Attention heads for the ligand cross-attention.
    max_seq_len : int, default 512
        Maximum supported protein length in residues.
    dropout : float, default 0.0
        Shared dropout probability.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_aa import (
    ...     RFDiffusionAADenoiser)
    >>> denoiser = RFDiffusionAADenoiser(
    ...     embed_dim=32, pair_dim=16, num_blocks=1, num_heads=4,
    ...     pair_num_heads=2, ligand_num_heads=4)
    >>> noisy_coords = torch.randn(2, 6, 9)
    >>> rotations = torch.eye(3).expand(2, 6, 3, 3).contiguous()
    >>> translations = torch.zeros(2, 6, 3)
    >>> t = torch.tensor([3, 7])
    >>> mask = torch.ones(2, 6)
    >>> ligand_types = torch.zeros(2, 4, dtype=torch.long)
    >>> ligand_coords = torch.randn(2, 4, 3)
    >>> pred_r, pred_t = denoiser([
    ...     noisy_coords, rotations, translations, t, mask,
    ...     ligand_types, ligand_coords, None])
    >>> pred_r.shape, pred_t.shape
    (torch.Size([2, 6, 3, 3]), torch.Size([2, 6, 3]))
    """

    def __init__(self,
                 embed_dim: int = 128,
                 pair_dim: int = 64,
                 time_dim: int = 128,
                 num_blocks: int = 2,
                 num_heads: int = 8,
                 pair_num_heads: int = 4,
                 ligand_num_heads: int = 8,
                 max_seq_len: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.coord_embed = ResidueEmbedding(9, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.ligand_embed = LigandContextEmbedding(embed_dim)
        self.ligand_cross_attn = BinderCrossAttention(
            embed_dim, num_heads=ligand_num_heads, dropout=dropout)
        self.stack = RFDiffusionMultiTrackStack(
            embed_dim=embed_dim,
            pair_dim=pair_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pair_num_heads=pair_num_heads,
            dropout=dropout,
        )

    def forward(
        self, inputs: List[Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict denoised backbone frames conditioned on a fixed ligand.

        Parameters
        ----------
        inputs : list
            ``[noisy_coords, rotations, translations, t, mask,
            ligand_atom_types, ligand_coords, ligand_mask]``. The first
            five entries match
            :class:`~deepchem.models.torch_models.rfdiffusion_multitrack.RFDiffusionMultiTrackDenoiser`.
            ``ligand_atom_types`` (``(batch, num_ligand_atoms)`` long)
            and ``ligand_coords`` (``(batch, num_ligand_atoms, 3)``) may
            be ``None`` to run unconditioned (matching the plain
            backbone denoiser); ``ligand_mask`` (``(batch,
            num_ligand_atoms)``) is optional even when a ligand is
            given.

        Returns
        -------
        pred_rotations : torch.Tensor
            Predicted denoised rotations, shape ``(batch, num_residues,
            3, 3)``.
        pred_translations : torch.Tensor
            Predicted denoised translations, shape ``(batch,
            num_residues, 3)``.
        """
        (noisy_coords, rotations, translations, t, mask, ligand_atom_types,
         ligand_coords, ligand_mask) = inputs
        t = t.long()
        t_emb = self.time_mlp(self.time_embedding(t))
        single = self.pos_encoding(self.coord_embed(noisy_coords))

        if ligand_atom_types is not None and ligand_coords is not None:
            ligand_feat = self.ligand_embed(ligand_atom_types, ligand_coords)
            single = single + self.ligand_cross_attn(
                single, ligand_feat, partner_mask=ligand_mask)

        attn_mask = mask.bool() if mask is not None else None
        _, _, pred_rotations, pred_translations = self.stack.forward_tracks(
            single,
            t_emb,
            attention_mask=attn_mask,
            rotations=rotations,
            translations=translations,
        )
        return pred_rotations, pred_translations


class RFDiffusionAA(TorchModel):
    """Ligand-conditioned RFDiffusion protein backbone model.

    Trains and samples protein backbones that are generated in the
    context of a single fixed ligand (e.g. designing a binding pocket
    or scaffold around a small molecule), using
    :class:`RFDiffusionAADenoiser`. Training and sampling otherwise
    follow the same combined IGSO(3) + translational diffusion process
    as ``RFDiffusionModel(architecture='multitrack')``: the network
    predicts the denoised frame directly at every step, and the loss is
    :func:`~deepchem.models.torch_models.rfdiffusion_multitrack.multitrack_frame_loss`
    plus a protein-ligand clash penalty term (see ``_ligand_protein_clash``
    below) that discourages generated backbone atoms from overlapping the
    ligand.

    See the module docstring for why this is ligand-conditioned
    *backbone* generation rather than full joint protein/ligand
    all-atom diffusion.

    Parameters
    ----------
    ligand : LigandPointCloud
        The fixed ligand every generated structure is conditioned on.
        Its coordinates are re-centered to the ligand's own centroid on
        construction so they sit in a comparable coordinate frame to
        the (per-batch centered) protein backbone.
    embed_dim : int, default 128
        Single-track channel size.
    pair_dim : int, default 64
        Pair-track channel size.
    time_dim : int, default 128
        Dimension of the sinusoidal timestep embedding.
    num_blocks : int, default 2
        Number of stacked multi-track blocks.
    num_heads : int, default 8
        Attention heads for the single track and IPA.
    pair_num_heads : int, default 4
        Attention heads for triangular self-attention.
    ligand_num_heads : int, default 8
        Attention heads for the ligand cross-attention.
    num_diffusion_steps : int, default 1000
        Total number of diffusion timesteps T.
    max_seq_len : int, default 512
        Maximum protein length in residues the model can handle.
    dropout : float, default 0.1
        Dropout probability used in the denoiser blocks.
    batch_size : int, default 4
        Number of proteins per training batch.
    learning_rate : float, default 1e-4
        Learning rate passed to the Adam optimizer.
    rotation_weight : float, default 1.0
        Weight on the rotation term of the frame loss.
    clash_weight : float, default 1.0
        Weight on the ligand clash loss term.
    clash_tolerance : float, default 1.5
        Distance (Angstrom) below which a generated CA atom and a
        ligand atom count as clashing. Looser than a typical all-atom
        van der Waals tolerance since only CA positions (not full
        backbone/side-chain atoms) are available to check.
    so3_beta_min : float, default 0.1
        Smallest IGSO(3) sigma (at t=1).
    so3_beta_max : float, default 1.5
        Largest IGSO(3) sigma (at t=T).
    device : torch.device, optional
        Device to train and sample on. Defaults to GPU if available,
        otherwise CPU.
    **kwargs
        Additional keyword arguments forwarded to ``TorchModel``.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.utils.rfdiffusion_ligand import LigandPointCloud
    >>> from deepchem.models.torch_models.rfdiffusion_aa import RFDiffusionAA
    >>> ligand = LigandPointCloud(
    ...     coords=np.random.randn(4, 3).astype(np.float32),
    ...     atom_types=np.array([4, 12, 12, 8]),
    ...     atomic_numbers=np.array([8, 6, 6, 7]),
    ...     bond_features=np.zeros((4, 4), dtype=np.int64))
    >>> proteins = [np.random.randn(15, 9).astype(np.float32)
    ...             for _ in range(4)]
    >>> X = np.empty(4, dtype=object)
    >>> for i, p in enumerate(proteins):
    ...     X[i] = p
    >>> dataset = dc.data.NumpyDataset(X=X, y=np.zeros((4, 1),
    ...                                                 dtype=np.float32))
    >>> model = RFDiffusionAA(
    ...     ligand=ligand, embed_dim=32, pair_dim=16, num_blocks=1,
    ...     num_heads=4, pair_num_heads=2, ligand_num_heads=4,
    ...     num_diffusion_steps=10, batch_size=2)
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> samples = model.generate(num_samples=2, seq_length=15)
    >>> samples.shape
    (2, 15, 9)
    """

    def __init__(self,
                 ligand: LigandPointCloud,
                 embed_dim: int = 128,
                 pair_dim: int = 64,
                 time_dim: int = 128,
                 num_blocks: int = 2,
                 num_heads: int = 8,
                 pair_num_heads: int = 4,
                 ligand_num_heads: int = 8,
                 num_diffusion_steps: int = 1000,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 batch_size: int = 4,
                 learning_rate: float = 1e-4,
                 rotation_weight: float = 1.0,
                 clash_weight: float = 1.0,
                 clash_tolerance: float = 1.5,
                 so3_beta_min: float = 0.1,
                 so3_beta_max: float = 1.5,
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if num_diffusion_steps <= 0:
            raise ValueError('num_diffusion_steps must be positive.')
        if max_seq_len <= 0:
            raise ValueError('max_seq_len must be positive.')

        self.num_diffusion_steps = num_diffusion_steps
        self.max_seq_len = max_seq_len
        self.coord_dim = 9
        self.rotation_weight = rotation_weight
        self.clash_weight = clash_weight
        self.clash_tolerance = clash_tolerance
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[float] = None

        ligand_center = ligand.coords.mean(axis=0, keepdims=True)
        self._ligand_atom_types = torch.tensor(ligand.atom_types,
                                               dtype=torch.long)
        self._ligand_coords = torch.tensor(ligand.coords - ligand_center,
                                           dtype=torch.float32)

        self.schedule = CosineSchedule(num_timesteps=num_diffusion_steps)
        self.igso3 = IGSO3(
            log_beta_schedule(num_diffusion_steps,
                              beta_min=so3_beta_min,
                              beta_max=so3_beta_max))

        backbone = RFDiffusionAADenoiser(
            embed_dim=embed_dim,
            pair_dim=pair_dim,
            time_dim=time_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pair_num_heads=pair_num_heads,
            ligand_num_heads=ligand_num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        backbone.schedule = self.schedule  # type: ignore[assignment]

        def loss_fn(outputs: List[torch.Tensor], labels: List[torch.Tensor],
                    weights: List[torch.Tensor]) -> torch.Tensor:
            frame_loss = multitrack_frame_loss(
                outputs, labels, weights, rotation_weight=self.rotation_weight)
            pred_translations = outputs[1]
            mask = weights[0]
            ligand_coords_batch = self._ligand_coords.to(
                pred_translations.device).unsqueeze(0).expand(
                    pred_translations.shape[0], -1, -1)
            ligand_radii = torch.full((ligand_coords_batch.shape[1],),
                                      1.7,
                                      device=pred_translations.device)
            protein_radii = torch.full((pred_translations.shape[1],),
                                       1.7,
                                       device=pred_translations.device)
            clash = _ligand_protein_clash(pred_translations,
                                          protein_radii,
                                          ligand_coords_batch,
                                          ligand_radii,
                                          mask,
                                          tolerance=self.clash_tolerance)
            return frame_loss + self.clash_weight * clash

        super(RFDiffusionAA, self).__init__(backbone,
                                            loss=loss_fn,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            device=device,
                                            **kwargs)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Center and scale backbone coordinates, matching ``RFDiffusionModel``.

        Parameters
        ----------
        coords : np.ndarray
            Backbone coordinates with shape ``(L, 3, 3)`` or ``(L, 9)``.

        Returns
        -------
        np.ndarray
            Normalized coordinates of shape ``(L, 9)``, dtype
            ``float32``.
        """
        if coords.ndim == 3 and coords.shape[1] == 3 and coords.shape[2] == 3:
            coords = coords.reshape(-1, 9)
        elif coords.ndim != 2 or coords.shape[1] != self.coord_dim:
            raise ValueError('coords must have shape (L, 3, 3) or (L, 9).')
        if coords.shape[0] == 0:
            raise ValueError('coords must have at least one residue.')
        ca_coords = coords[:, 3:6]
        centroid = ca_coords.mean(axis=0, keepdims=True)
        coords = coords - np.tile(centroid, 3)
        std = coords.std()
        if std > 1e-6:
            coords = coords / std
        return coords.astype(np.float32)

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """Yield ligand-conditioned diffusion training batches.

        Parameters
        ----------
        dataset : Dataset
            DeepChem dataset whose ``X`` entries are backbone coordinate
            arrays of shape ``(L, 9)`` or ``(L, 3, 3)``.
        epochs : int, default 1
            Number of passes over the dataset.
        mode : str, default 'fit'
            Only ``'fit'`` is supported.
        deterministic : bool, default True
            Whether to iterate over the dataset in a fixed order.
        pad_batches : bool, default True
            Whether to pad the last batch to ``batch_size``.

        Yields
        ------
        tuple
            ``([noisy_coords, noisy_rotations, noisy_translations,
            timesteps, mask, ligand_atom_types, ligand_coords,
            ligand_mask], [rotations0, translations0], [weights])``,
            matching ``RFDiffusionModel``'s multitrack batch format with
            the fixed ligand tensors appended. ``ligand_mask`` is always
            all-ones here since the conditioning ligand is fully
            specified (no padding).

        Raises
        ------
        NotImplementedError
            If ``mode`` is not ``'fit'``.
        ValueError
            If any protein's length exceeds ``max_seq_len``.
        """
        if mode != 'fit':
            raise NotImplementedError(
                'RFDiffusionAA does not support predict/uncertainty mode. '
                'Use generate() instead.')

        for _epoch in range(epochs):
            for (X_b, _y_b, w_b,
                 _ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                deterministic=deterministic,
                                                pad_batches=pad_batches):
                batch_size = len(X_b)
                sample_weights = np.asarray(w_b, dtype=np.float32)
                if sample_weights.ndim == 0:
                    sample_weights = np.full((batch_size,),
                                             float(sample_weights),
                                             dtype=np.float32)
                else:
                    sample_weights = sample_weights.reshape(
                        batch_size, -1).max(axis=1).astype(np.float32)

                normalized = []
                lengths = []
                for i in range(batch_size):
                    coords = X_b[i]
                    if isinstance(coords, np.ndarray) and coords.size > 0:
                        c = self._normalize_coords(coords)
                        normalized.append(c)
                        lengths.append(c.shape[0])
                    else:
                        normalized.append(
                            np.zeros((1, self.coord_dim), dtype=np.float32))
                        lengths.append(1)

                max_len = max(lengths)
                if max_len > self.max_seq_len:
                    raise ValueError(
                        f'Protein length {max_len} exceeds max_seq_len '
                        f'{self.max_seq_len}. Increase max_seq_len or crop.')

                all_raw = [
                    X_b[i].reshape(-1, 9) if X_b[i].ndim == 3 else X_b[i]
                    for i in range(batch_size)
                    if (sample_weights[i] > 0 and
                        isinstance(X_b[i], np.ndarray) and X_b[i].size > 0)
                ]
                if all_raw:
                    raw = np.concatenate(all_raw, axis=0)
                    centroid = raw[:, 3:6].mean(axis=0, keepdims=True)
                    centered = raw - np.tile(centroid, 3)
                    batch_std = float(centered.std())
                    if self._train_std is None:
                        self._train_mean = centroid[0]
                        self._train_std = batch_std
                    else:
                        alpha = 0.1
                        assert self._train_mean is not None
                        self._train_mean = ((1 - alpha) * self._train_mean +
                                            alpha * centroid[0])
                        self._train_std = ((1 - alpha) * self._train_std +
                                           alpha * batch_std)

                batch_coords = []
                batch_masks = []
                for c in normalized:
                    padded = np.zeros((max_len, self.coord_dim),
                                      dtype=np.float32)
                    padded[:c.shape[0]] = c
                    batch_coords.append(padded)
                    mask = np.zeros((max_len,), dtype=np.float32)
                    mask[:c.shape[0]] = 1.0
                    batch_masks.append(mask)

                coords_batch = np.stack(batch_coords, axis=0)
                mask_batch = np.stack(batch_masks, axis=0)
                t = np.random.randint(0,
                                      self.num_diffusion_steps,
                                      size=(batch_size,))
                coords_tensor = torch.tensor(coords_batch, dtype=torch.float32)
                t_tensor = torch.tensor(t, dtype=torch.long)
                mask_tensor = torch.tensor(mask_batch, dtype=torch.float32)

                rotations0, translations0 = build_backbone_frames(
                    coords_tensor.reshape(batch_size, max_len, 3, 3))
                identity = torch.eye(
                    3, dtype=rotations0.dtype).expand_as(rotations0)
                valid = mask_tensor.bool()
                rotations0 = torch.where(valid[..., None, None], rotations0,
                                         identity)
                translations0 = translations0 * mask_tensor[..., None]

                noisy_rotations, noisy_translations = sample_noisy_frames(
                    self.igso3, self.schedule, rotations0, translations0,
                    t_tensor)
                noisy_coords_9 = backbone_coords_from_frames(
                    noisy_rotations,
                    noisy_translations).reshape(batch_size, max_len, 9)

                ligand_types_batch = self._ligand_atom_types.unsqueeze(
                    0).expand(batch_size, -1).numpy()
                ligand_coords_batch = self._ligand_coords.unsqueeze(0).expand(
                    batch_size, -1, -1).numpy()
                ligand_mask_batch = np.ones(
                    (batch_size, ligand_types_batch.shape[1]), dtype=np.float32)

                weights_mask = mask_batch * sample_weights[:, None]

                yield ([
                    noisy_coords_9.numpy(),
                    noisy_rotations.numpy(),
                    noisy_translations.numpy(),
                    t.astype(np.int64), mask_batch, ligand_types_batch,
                    ligand_coords_batch, ligand_mask_batch
                ], [rotations0.numpy(),
                    translations0.numpy()], [weights_mask])

    def generate(self,
                 num_samples: int = 1,
                 seq_length: int = 50,
                 device: Optional[torch.device] = None) -> np.ndarray:
        """Generate protein backbones conditioned on the fixed ligand.

        Parameters
        ----------
        num_samples : int, default 1
            Number of backbone structures to generate.
        seq_length : int, default 50
            Number of residues in each generated structure.
        device : torch.device, optional
            Device to run sampling on. Defaults to the model's current
            device.

        Returns
        -------
        np.ndarray
            Generated backbone coordinates of shape
            ``(num_samples, seq_length, 9)``, dtype float32.

        Raises
        ------
        ValueError
            If ``num_samples <= 0``, ``seq_length <= 0``, or
            ``seq_length > max_seq_len``.
        """
        if num_samples <= 0:
            raise ValueError('num_samples must be positive.')
        if seq_length <= 0:
            raise ValueError('seq_length must be positive.')
        if seq_length > self.max_seq_len:
            raise ValueError(
                f'seq_length {seq_length} exceeds max_seq_len {self.max_seq_len}.'
            )
        if device is None:
            device = self.device

        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                rotations = self.igso3.sample(
                    sigma_index=self.num_diffusion_steps - 1,
                    shape=(num_samples, seq_length)).to(device)
                translations = torch.randn(num_samples,
                                           seq_length,
                                           3,
                                           device=device)
                mask = torch.ones(num_samples, seq_length, device=device)
                ligand_types = self._ligand_atom_types.to(device).unsqueeze(
                    0).expand(num_samples, -1)
                ligand_coords = self._ligand_coords.to(device).unsqueeze(
                    0).expand(num_samples, -1, -1)
                ligand_mask = torch.ones(num_samples,
                                         ligand_types.shape[1],
                                         device=device)

                for step in reversed(range(self.num_diffusion_steps)):
                    t_batch = torch.full((num_samples,),
                                         step,
                                         dtype=torch.long,
                                         device=device)
                    noisy_coords = backbone_coords_from_frames(
                        rotations,
                        translations).reshape(num_samples, seq_length, 9)
                    pred_rotations, pred_translations = self.model([
                        noisy_coords, rotations, translations, t_batch, mask,
                        ligand_types, ligand_coords, ligand_mask
                    ])
                    if step > 0:
                        rotations = so3_x0_reverse_step(self.igso3, rotations,
                                                        pred_rotations, step)
                        translations = translation_posterior_step(
                            self.schedule, pred_translations, translations,
                            step)
                    else:
                        rotations, translations = pred_rotations, pred_translations

                samples = backbone_coords_from_frames(
                    rotations, translations).reshape(num_samples, seq_length, 9)
        finally:
            self.model.train(was_training)

        result = samples.cpu().numpy()
        if self._train_std is not None and self._train_std > 1e-6:
            result = result * self._train_std
        if self._train_mean is not None:
            result = result + np.tile(self._train_mean, 3)
        return result

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir: Optional[str] = None) -> None:
        """Save model weights and training normalization statistics."""
        super().save_checkpoint(max_checkpoints_to_keep=max_checkpoints_to_keep,
                                model_dir=model_dir)
        if max_checkpoints_to_keep == 0:
            return
        checkpoint = sorted(self.get_checkpoints(model_dir))[0]
        data = torch.load(checkpoint, map_location=self.device)
        data['rf_diffusion_aa_train_mean'] = (None if self._train_mean is None
                                              else self._train_mean.tolist())
        data['rf_diffusion_aa_train_std'] = self._train_std
        torch.save(data, checkpoint)

    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Restore model weights and training normalization statistics."""
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if not checkpoints:
                raise ValueError('No checkpoint found.')
            checkpoint = checkpoints[0]
        super().restore(checkpoint=checkpoint,
                        model_dir=model_dir,
                        strict=strict)
        data = torch.load(checkpoint, map_location=self.device)
        train_mean = data.get('rf_diffusion_aa_train_mean')
        self._train_mean = (None if train_mean is None else np.asarray(
            train_mean, dtype=np.float32))
        self._train_std = data.get('rf_diffusion_aa_train_std')


def _ligand_protein_clash(protein_coords: torch.Tensor,
                          protein_radii: torch.Tensor,
                          ligand_coords: torch.Tensor,
                          ligand_radii: torch.Tensor,
                          protein_mask: torch.Tensor,
                          tolerance: float = 1.5,
                          eps: float = 1e-8) -> torch.Tensor:
    """Batched protein-ligand clash penalty.

    Same steric-overlap penalty as
    :func:`~deepchem.models.torch_models.rfdiffusion_losses.ligand_clash_loss`,
    but that function penalizes every atom pair in a single point cloud
    regardless of which molecule each atom belongs to, which isn't what
    is needed here. This concatenates each batch element's protein and
    ligand atoms into one point cloud but masks out protein-protein and
    ligand-ligand pairs, so only cross (protein, ligand) pairs are
    penalized.

    Parameters
    ----------
    protein_coords : torch.Tensor
        Shape ``(batch, L, 3)``.
    protein_radii : torch.Tensor
        Shape ``(L,)``.
    ligand_coords : torch.Tensor
        Shape ``(batch, num_ligand_atoms, 3)``.
    ligand_radii : torch.Tensor
        Shape ``(num_ligand_atoms,)``.
    protein_mask : torch.Tensor
        Shape ``(batch, L)``.
    tolerance : float, default 1.5
    eps : float, default 1e-8

    Returns
    -------
    torch.Tensor
        Scalar clash penalty averaged over the batch.
    """
    batch, num_protein = protein_coords.shape[:2]
    num_ligand = ligand_coords.shape[1]

    all_coords = torch.cat([protein_coords, ligand_coords], dim=1)
    all_radii = torch.cat([
        protein_radii.unsqueeze(0).expand(batch, -1),
        ligand_radii.unsqueeze(0).expand(batch, -1)
    ],
                          dim=1)
    ligand_mask = torch.ones(batch, num_ligand, device=protein_mask.device)
    all_mask = torch.cat([protein_mask, ligand_mask], dim=1)

    n = num_protein + num_ligand
    diff = all_coords.unsqueeze(-2) - all_coords.unsqueeze(-3)
    dist = diff.norm(dim=-1).clamp(min=eps)
    radii_sum = all_radii.unsqueeze(-1) + all_radii.unsqueeze(-2)
    violation = (radii_sum - tolerance - dist).clamp(min=0.0)

    cross_only = torch.zeros(n, n, dtype=torch.bool, device=diff.device)
    cross_only[:num_protein, num_protein:] = True
    cross_only[num_protein:, :num_protein] = True
    pair_mask = cross_only.to(dist.dtype).unsqueeze(0) * \
        all_mask.unsqueeze(-1) * all_mask.unsqueeze(-2)

    denom = pair_mask.sum(dim=(-2, -1)).clamp(min=eps)
    return (((violation**2) * pair_mask).sum(dim=(-2, -1)) / denom).mean()
