"""RFDiffusionModel: TorchModel wrapper for protein backbone diffusion.

This module adds ``RFDiffusionModel``, a DeepChem ``TorchModel`` that lets
you train and sample protein backbone structures using standard DeepChem
workflows. Two denoiser architectures are available, selected with the
``architecture`` argument:

- ``'multitrack'`` (default) -- the actual RFdiffusion architecture: rigid
  backbone frames, Invariant Point Attention, the pair track, and the
  sequence/structure track (``RFDiffusionMultiTrackStack``), combined with
  IGSO(3) rotational diffusion and Gaussian translational diffusion. This
  is the scientifically faithful path, and the only one that supports
  contig/motif conditioning.
- ``'transformer'`` -- the lightweight ``BackboneDiffusion`` baseline: a
  plain Transformer denoiser trained with translation-only DDPM on flat
  ``(N, CA, C)`` coordinates. Kept as a cheap, quick-to-train fallback;
  pass ``architecture='transformer'`` to use it.

The building-block layers (``SinusoidalTimestepEmbedding``,
``ResidueEmbedding``, ``PositionalEncoding``, ``CosineSchedule``,
``DiffusionTransformerBlock``, ``BackboneDiffusion``) live in
``deepchem.models.torch_models.layers``. The multi-track pieces
(``RFDiffusionMultiTrackDenoiser`` and the diffusion helpers around it)
live in ``deepchem.models.torch_models.rfdiffusion_multitrack``.

Classes
-------
- ``RFDiffusionModel`` -- TorchModel wrapper that handles training batches,
  normalization, sampling, checkpointing, and restore.

References
----------
.. [1] Watson, J. L., et al. "De novo design of protein structure and function
   with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
.. [2] Ho, J., Jain, A., & Abbeel, P. "Denoising diffusion probabilistic
   models." NeurIPS 2020.
.. [3] Nichol, A. Q., & Dhariwal, P. "Improved denoising diffusion
   probabilistic models." ICML 2021.

Notes
-----
This module requires PyTorch to be installed.
"""

import logging

import numpy as np
from typing import Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('RFDiffusionModel requires PyTorch to be installed.')

from deepchem.data import Dataset
from deepchem.models.torch_models.layers import BackboneDiffusion, CosineSchedule
from deepchem.models.torch_models.rfdiffusion_frames import (
    build_backbone_frames,)
from deepchem.models.torch_models.rfdiffusion_multitrack import (
    RFDiffusionMultiTrackDenoiser,
    backbone_coords_from_frames,
    multitrack_frame_loss,
    sample_noisy_frames,
    so3_x0_reverse_step,
    translation_posterior_step,
)
from deepchem.models.torch_models.rfdiffusion_so3 import IGSO3, log_beta_schedule
from deepchem.models.torch_models.torch_model import TorchModel

logger = logging.getLogger(__name__)


def _diffusion_loss(outputs: List[torch.Tensor], labels: List[torch.Tensor],
                    weights: List[torch.Tensor]) -> torch.Tensor:
    """MSE loss between predicted noise and true noise, with masking.

    Parameters
    ----------
    outputs : list of torch.Tensor
        Model outputs; ``outputs[0]`` is the predicted noise tensor of shape
        ``(batch, seq_len, coord_dim)``.
    labels : list of torch.Tensor
        Ground-truth noise; ``labels[0]`` has the same shape as ``outputs[0]``.
    weights : list of torch.Tensor
        Per-element weights of shape ``(batch, seq_len, coord_dim)`` that mask out
        padded positions and scale by sample weight.

    Returns
    -------
    torch.Tensor
        Scalar weighted-MSE loss.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion import _diffusion_loss
    >>> pred = [torch.zeros(2, 5, 9)]
    >>> true = [torch.ones(2, 5, 9)]
    >>> weights = [torch.ones(2, 5, 9)]
    >>> loss = _diffusion_loss(pred, true, weights)
    >>> float(loss)
    1.0
    """
    pred_noise = outputs[0]
    true_noise = labels[0]
    w = weights[0]
    loss = F.mse_loss(pred_noise, true_noise, reduction='none')
    if w.dim() < loss.dim():
        w = w.reshape(w.shape + (1,) * (loss.dim() - w.dim()))
    denom = torch.clamp(w.expand_as(loss).sum(), min=1.0)
    return (loss * w).sum() / denom


class RFDiffusionModel(TorchModel):
    """TorchModel wrapper for the RFDiffusion protein backbone model.

    Wraps a backbone denoiser in DeepChem's ``TorchModel`` interface so you
    can train with ``model.fit(dataset)`` and generate new backbone
    structures with ``model.generate()``. Two denoisers are available via
    ``architecture``:

    - ``'multitrack'`` (default): the real RFdiffusion network
      (``RFDiffusionMultiTrackDenoiser``), trained to predict the denoised
      backbone frame (rotation + translation) directly at every step.
      Rotations are diffused with IGSO(3) and translations with a
      Gaussian DDPM process; the training loss is a masked geodesic
      rotation loss plus a masked translation MSE
      (``multitrack_frame_loss``). This is the only architecture that
      supports contig/motif conditioning in ``generate()``.
    - ``'transformer'``: the lightweight ``BackboneDiffusion`` baseline.
      Training uses the standard DDPM noise-prediction objective: at each
      step a random timestep is sampled, Gaussian noise is added to the
      clean coordinates with ``CosineSchedule.q_sample``, and the model
      learns to predict the added noise via an MSE loss. Cheaper to train
      than ``'multitrack'``, useful for quick experimentation.

    Coordinates are center-normalized before being fed to the model and
    denormalized when samples are returned from ``generate()``.

    Parameters
    ----------
    embed_dim : int, default 256
        Hidden (single-track) dimension for the denoiser.
    time_dim : int, default 128
        Dimension of the sinusoidal timestep embedding.
    num_layers : int, default 8
        Number of ``DiffusionTransformerBlock`` layers. Only used when
        ``architecture='transformer'``.
    num_heads : int, default 8
        Number of attention heads per block.
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
    device : torch.device, optional
        Device to train and sample on. Defaults to GPU if available,
        otherwise CPU.
    architecture : str, default 'multitrack'
        Either ``'multitrack'`` or ``'transformer'``.
    pair_dim : int, default 64
        Pair-track channel size. Only used when
        ``architecture='multitrack'``.
    num_blocks : int, default 2
        Number of stacked multi-track blocks. Only used when
        ``architecture='multitrack'``.
    pair_num_heads : int, default 4
        Attention heads for triangular self-attention. Only used when
        ``architecture='multitrack'``.
    rotation_weight : float, default 1.0
        Weight on the rotation term of the multi-track training loss.
        Only used when ``architecture='multitrack'``.
    so3_beta_min : float, default 0.1
        Smallest IGSO(3) sigma (at t=1). Only used when
        ``architecture='multitrack'``.
    so3_beta_max : float, default 1.5
        Largest IGSO(3) sigma (at t=T). Only used when
        ``architecture='multitrack'``.
    **kwargs
        Additional keyword arguments forwarded to ``TorchModel``.

    Raises
    ------
    ValueError
        If `embed_dim`, `time_dim`, `num_layers`, `num_heads`,
        `num_diffusion_steps`, or `max_seq_len` is not positive, or if
        `architecture` is not `'transformer'` or `'multitrack'`.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> proteins = [np.random.randn(20, 9).astype(np.float32) for _ in range(6)]
    >>> X = np.empty(6, dtype=object)
    >>> for i, p in enumerate(proteins):
    ...     X[i] = p
    >>> dataset = dc.data.NumpyDataset(X=X, y=np.zeros((6, 1), dtype=np.float32))
    >>> model = dc.models.RFDiffusionModel(
    ...     embed_dim=32, pair_dim=16, num_blocks=1, num_heads=4,
    ...     pair_num_heads=2, num_diffusion_steps=20, batch_size=2)
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> samples = model.generate(num_samples=2, seq_length=20)
    >>> samples.shape
    (2, 20, 9)

    The lightweight Transformer baseline is used the same way, by passing
    ``architecture='transformer'``:

    >>> baseline_model = dc.models.RFDiffusionModel(
    ...     architecture='transformer', embed_dim=64, num_layers=2,
    ...     num_heads=4, num_diffusion_steps=50, batch_size=2)
    >>> loss = baseline_model.fit(dataset, nb_epoch=1)
    >>> samples = baseline_model.generate(num_samples=2, seq_length=20)
    >>> samples.shape
    (2, 20, 9)

    References
    ----------
    .. [1] Watson, J. L., et al. "De novo design of protein structure and
       function with RFdiffusion." Nature 620.7976 (2023): 1089-1100.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 time_dim: int = 128,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 num_diffusion_steps: int = 1000,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 batch_size: int = 4,
                 learning_rate: float = 1e-4,
                 device: Optional[torch.device] = None,
                 architecture: str = 'multitrack',
                 pair_dim: int = 64,
                 num_blocks: int = 2,
                 pair_num_heads: int = 4,
                 rotation_weight: float = 1.0,
                 so3_beta_min: float = 0.1,
                 so3_beta_max: float = 1.5,
                 **kwargs) -> None:
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if time_dim <= 0:
            raise ValueError('time_dim must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive.')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive.')
        if architecture not in ('transformer', 'multitrack'):
            raise ValueError(
                f"architecture must be 'transformer' or 'multitrack', got "
                f"{architecture!r}.")
        if num_diffusion_steps <= 0:
            raise ValueError('num_diffusion_steps must be positive.')
        if max_seq_len <= 0:
            raise ValueError('max_seq_len must be positive.')

        self.num_diffusion_steps = num_diffusion_steps
        self.max_seq_len = max_seq_len
        self.coord_dim = 9
        self.architecture = architecture
        self.rotation_weight = rotation_weight
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[float] = None

        self.schedule = CosineSchedule(num_timesteps=num_diffusion_steps)

        if architecture == 'transformer':
            self.igso3: Optional[IGSO3] = None
            backbone: torch.nn.Module = BackboneDiffusion(
                coord_dim=self.coord_dim,
                embed_dim=embed_dim,
                time_dim=time_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            loss_fn = _diffusion_loss
        else:
            self.igso3 = IGSO3(
                log_beta_schedule(num_diffusion_steps,
                                  beta_min=so3_beta_min,
                                  beta_max=so3_beta_max))
            backbone = RFDiffusionMultiTrackDenoiser(
                embed_dim=embed_dim,
                pair_dim=pair_dim,
                time_dim=time_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                pair_num_heads=pair_num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )

            def loss_fn(outputs: List[torch.Tensor], labels: List[torch.Tensor],
                        weights: List[torch.Tensor]) -> torch.Tensor:
                return multitrack_frame_loss(outputs,
                                             labels,
                                             weights,
                                             rotation_weight=rotation_weight)

        # Attach schedule so it moves with the model when .to(device) is called
        backbone.schedule = self.schedule

        super(RFDiffusionModel, self).__init__(backbone,
                                               loss=loss_fn,
                                               batch_size=batch_size,
                                               learning_rate=learning_rate,
                                               device=device,
                                               **kwargs)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Center and scale backbone coordinates for diffusion training.

        Translates coordinates so that the CA-atom centroid is at the origin,
        then divides by the global standard deviation of all coordinates. This
        normalization keeps the diffusion prior (standard Gaussian) well-matched
        to the data distribution and is reversed in ``generate()``.

        Both ``(L, 3, 3)`` arrays (N/CA/C stacked along axis 1) and flat
        ``(L, 9)`` arrays are accepted.

        Parameters
        ----------
        coords : np.ndarray
            Backbone coordinates with shape ``(L, 3, 3)`` or ``(L, 9)``,
            where L is the number of residues.

        Returns
        -------
        np.ndarray
            Normalized coordinates of shape ``(L, 9)`` and dtype
            ``float32``.

        Raises
        ------
        ValueError
            If ``coords`` does not have a supported shape, or if it is empty.

        Examples
        --------
        >>> import numpy as np
        >>> import deepchem as dc
        >>> model = dc.models.RFDiffusionModel(
        ...     embed_dim=32, pair_dim=16, num_blocks=1,
        ...     num_diffusion_steps=10)
        >>> coords_33 = np.random.randn(10, 3, 3).astype(np.float32)
        >>> out = model._normalize_coords(coords_33)
        >>> out.shape
        (10, 9)
        >>> coords_9 = np.random.randn(10, 9).astype(np.float32)
        >>> out2 = model._normalize_coords(coords_9)
        >>> out2.shape
        (10, 9)
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

    def _pad_coords(self, coords: np.ndarray,
                    max_len: int) -> Tuple[np.ndarray, int]:
        """Pad a coordinate array to ``max_len`` with zeros.

        Used inside ``default_generator`` to form rectangular batches when
        proteins in the same batch have different lengths.

        Parameters
        ----------
        coords : np.ndarray
            Normalized backbone coordinates of shape ``(L, 9)``.
        max_len : int
            Target padded sequence length. Must be >= ``coords.shape[0]``.

        Returns
        -------
        padded : np.ndarray
            Zero-padded array of shape ``(max_len, 9)`` and dtype
            ``float32``.
        orig_len : int
            Original unpadded length ``L``.

        Raises
        ------
        ValueError
            If ``coords.shape[0] > max_len``.

        Examples
        --------
        >>> import numpy as np
        >>> import deepchem as dc
        >>> model = dc.models.RFDiffusionModel(
        ...     embed_dim=32, pair_dim=16, num_blocks=1,
        ...     num_diffusion_steps=10)
        >>> coords = np.random.randn(5, 9).astype(np.float32)
        >>> padded, orig_len = model._pad_coords(coords, 10)
        >>> padded.shape
        (10, 9)
        >>> orig_len
        5
        """
        orig_len = coords.shape[0]
        if orig_len > max_len:
            raise ValueError(
                f'Protein length {orig_len} exceeds target {max_len}.')
        padded = np.zeros((max_len, self.coord_dim), dtype=np.float32)
        padded[:orig_len] = coords
        return padded, orig_len

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """Yield diffusion training batches from a DeepChem dataset.

        For each mini-batch this method normalizes each protein's
        coordinates with ``_normalize_coords``, pads them to a common
        length with ``_pad_coords``, and samples random diffusion
        timesteps uniformly from ``[0, num_diffusion_steps)``. What gets
        yielded after that depends on ``architecture``:

        - ``'transformer'``: adds noise with ``CosineSchedule.q_sample``
          and yields ``([noisy_coords, timesteps, mask], [noise],
          [weights])``, where ``mask`` is a float array marking valid
          (non-padded) positions.
        - ``'multitrack'``: builds ground-truth rigid frames from the
          padded coordinates with ``build_backbone_frames``, diffuses them
          with ``sample_noisy_frames`` (IGSO(3) for rotations,
          ``CosineSchedule.q_sample`` for translations), and yields
          ``([noisy_coords, noisy_rotations, noisy_translations,
          timesteps, mask], [rotations0, translations0], [weights])``.

        Running statistics for CA-centroid and coordinate std are updated
        each batch and used by ``generate()`` to denormalize samples.

        Parameters
        ----------
        dataset : Dataset
            DeepChem dataset whose ``X`` entries are backbone coordinate
            arrays of shape ``(L, 9)`` or ``(L, 3, 3)``.
        epochs : int, default 1
            Number of passes over the dataset.
        mode : str, default 'fit'
            Only ``'fit'`` is supported. Use ``generate()`` for sampling.
        deterministic : bool, default True
            Whether to iterate over the dataset in a fixed order.
        pad_batches : bool, default True
            Whether to pad the last batch to ``batch_size`` if it is smaller.

        Yields
        ------
        tuple
            See the architecture-dependent formats described above.
            Shapes shared by both: ``noisy_coords`` is ``(batch, max_len,
            9)``; ``timesteps`` is ``(batch,)`` int64; ``mask`` is
            ``(batch, max_len)`` float32 (1.0 for valid positions, 0.0 for
            padding). For ``'transformer'``, ``noise`` and ``weights`` are
            ``(batch, max_len, 9)``. For ``'multitrack'``,
            ``noisy_rotations`` / ``rotations0`` are ``(batch, max_len, 3,
            3)``; ``noisy_translations`` / ``translations0`` are ``(batch,
            max_len, 3)``; ``weights`` is ``(batch, max_len)``.

        Raises
        ------
        NotImplementedError
            If ``mode`` is not ``'fit'``.
        ValueError
            If any protein's length exceeds ``max_seq_len``.

        Examples
        --------
        >>> import numpy as np
        >>> import deepchem as dc
        >>> proteins = [np.random.randn(10, 9).astype(np.float32)
        ...             for _ in range(4)]
        >>> X = np.empty(4, dtype=object)
        >>> for i, p in enumerate(proteins):
        ...     X[i] = p
        >>> dataset = dc.data.NumpyDataset(X=X, y=np.zeros((4, 1),
        ...                                                 dtype=np.float32))
        >>> model = dc.models.RFDiffusionModel(
        ...     embed_dim=32, num_layers=1, num_heads=4,
        ...     num_diffusion_steps=10, batch_size=2)
        >>> gen = model.default_generator(dataset, epochs=1)
        >>> inputs, labels, weights = next(gen)
        >>> inputs[0].shape  # noisy_coords
        (2, 10, 9)
        """
        if mode != 'fit':
            raise NotImplementedError(
                'RFDiffusionModel does not support predict/uncertainty mode. '
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

                # Update running normalization statistics for denormalization
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
                    padded, orig_len = self._pad_coords(c, max_len)
                    batch_coords.append(padded)
                    mask = np.zeros((max_len,), dtype=np.float32)
                    mask[:orig_len] = 1.0
                    batch_masks.append(mask)

                coords_batch = np.stack(batch_coords, axis=0)
                mask_batch = np.stack(batch_masks, axis=0)
                t = np.random.randint(0,
                                      self.num_diffusion_steps,
                                      size=(batch_size,))
                coords_tensor = torch.tensor(coords_batch, dtype=torch.float32)
                t_tensor = torch.tensor(t, dtype=torch.long)

                if self.architecture == 'transformer':
                    noisy_coords, noise = self.schedule.q_sample(
                        coords_tensor, t_tensor)

                    noisy_np = noisy_coords.numpy()
                    noise_np = noise.numpy()
                    t_np = t.astype(np.int64)
                    weights = (mask_batch[:, :, None].astype(np.float32) *
                               sample_weights[:, None, None])

                    yield ([noisy_np, t_np, mask_batch], [noise_np], [weights])
                else:
                    mask_tensor = torch.tensor(mask_batch, dtype=torch.float32)
                    rotations0, translations0 = build_backbone_frames(
                        coords_tensor.reshape(batch_size, max_len, 3, 3))
                    # Padded positions have all-zero (N, CA, C) coordinates,
                    # which build_backbone_frames cannot turn into a valid
                    # orthonormal frame. Replace those with an identity
                    # frame so they cannot introduce NaNs into the network
                    # before the loss masks them out.
                    identity = torch.eye(
                        3, dtype=rotations0.dtype).expand_as(rotations0)
                    valid = mask_tensor.bool()
                    rotations0 = torch.where(valid[..., None, None], rotations0,
                                             identity)
                    translations0 = translations0 * mask_tensor[..., None]

                    assert self.igso3 is not None
                    noisy_rotations, noisy_translations = sample_noisy_frames(
                        self.igso3, self.schedule, rotations0, translations0,
                        t_tensor)
                    noisy_coords_9 = backbone_coords_from_frames(
                        noisy_rotations,
                        noisy_translations).reshape(batch_size, max_len, 9)

                    weights_mask = mask_batch * sample_weights[:, None]

                    yield ([
                        noisy_coords_9.numpy(),
                        noisy_rotations.numpy(),
                        noisy_translations.numpy(),
                        t.astype(np.int64), mask_batch
                    ], [rotations0.numpy(),
                        translations0.numpy()], [weights_mask])

    def generate(self,
                 num_samples: int = 1,
                 seq_length: int = 50,
                 device: Optional[torch.device] = None,
                 motif_mask: Optional[np.ndarray] = None,
                 motif_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate new protein backbone structures by running reverse diffusion.

        With ``architecture='transformer'``, starts from pure Gaussian noise
        of shape ``(num_samples, seq_length, 9)`` and iteratively denoises
        it over ``num_diffusion_steps`` steps using ``CosineSchedule.sample``.

        With ``architecture='multitrack'``, starts from an IGSO(3) sample
        for the rotations and Gaussian noise for the translations, and
        alternates ``so3_x0_reverse_step`` / ``translation_posterior_step``
        using the network's per-step frame prediction, following
        RFdiffusion's own reverse sampling process. This is also the only
        architecture that supports motif conditioning via ``motif_mask`` /
        ``motif_coords``: positions flagged in ``motif_mask`` are held at
        their reference frame throughout sampling, matching the
        ``diffusion_mask`` behavior in the original ``Diffuser.diffuse_pose``.
        Use :func:`~deepchem.utils.rfdiffusion_contigs.parse_contig_string`
        and :func:`~deepchem.utils.rfdiffusion_contigs.build_motif_mask` to
        build ``motif_mask`` / ``motif_coords`` from a contig string and a
        reference structure.

        If the model has been trained, the output is denormalized using the
        running CA-centroid and coordinate std tracked during training.

        Parameters
        ----------
        num_samples : int, default 1
            Number of backbone structures to generate.
        seq_length : int, default 50
            Number of residues in each generated structure.
        device : torch.device, optional
            Device to run sampling on. Defaults to the model's current device.
        motif_mask : np.ndarray, optional
            Boolean array of shape ``(seq_length,)``; ``True`` marks
            positions held fixed at ``motif_coords`` throughout sampling.
            Only valid with ``architecture='multitrack'``, and the same
            mask/coords are applied to every sample in the batch.
        motif_coords : np.ndarray, optional
            Reference ``(N, CA, C)`` coordinates of shape
            ``(seq_length, 3, 3)`` used at the positions flagged by
            ``motif_mask``. Must be passed together with ``motif_mask``.

        Returns
        -------
        np.ndarray
            Generated backbone coordinates of shape
            ``(num_samples, seq_length, 9)``, dtype float32.
            Columns 0-2 are N atom coords, 3-5 are CA, 6-8 are C.

        Raises
        ------
        ValueError
            If ``num_samples <= 0``, ``seq_length <= 0``,
            ``seq_length > max_seq_len``, if only one of ``motif_mask`` /
            ``motif_coords`` is given, if their shapes are inconsistent
            with ``seq_length``, or if motif conditioning is requested
            with ``architecture='transformer'``.

        Examples
        --------
        >>> import numpy as np
        >>> import deepchem as dc
        >>> model = dc.models.RFDiffusionModel(
        ...     embed_dim=32, pair_dim=16, num_blocks=1, num_heads=4,
        ...     pair_num_heads=2, num_diffusion_steps=10)
        >>> samples = model.generate(num_samples=2, seq_length=15)
        >>> samples.shape
        (2, 15, 9)
        >>> bool(np.isfinite(samples).all())
        True

        Motif-conditioned generation holds a reference sub-structure fixed
        while the rest is generated:

        >>> mt_model = dc.models.RFDiffusionModel(
        ...     embed_dim=32, pair_dim=16,
        ...     num_blocks=1, num_heads=4, pair_num_heads=2,
        ...     num_diffusion_steps=10)
        >>> mask = np.zeros(12, dtype=bool)
        >>> mask[4:8] = True
        >>> ref = np.random.randn(12, 3, 3).astype(np.float32)
        >>> samples = mt_model.generate(num_samples=1, seq_length=12,
        ...                             motif_mask=mask, motif_coords=ref)
        >>> samples.shape
        (1, 12, 9)
        """
        if num_samples <= 0:
            raise ValueError('num_samples must be positive.')
        if seq_length <= 0:
            raise ValueError('seq_length must be positive.')
        if seq_length > self.max_seq_len:
            raise ValueError(
                f'seq_length {seq_length} exceeds max_seq_len {self.max_seq_len}.'
            )
        if (motif_mask is None) != (motif_coords is None):
            raise ValueError(
                'motif_mask and motif_coords must be provided together.')
        if motif_mask is not None:
            if self.architecture != 'multitrack':
                raise ValueError(
                    "motif conditioning requires architecture='multitrack'.")
            if motif_mask.shape != (seq_length,):
                raise ValueError(
                    f'motif_mask must have shape ({seq_length},), got '
                    f'{motif_mask.shape}.')
            assert motif_coords is not None
            if motif_coords.shape != (seq_length, 3, 3):
                raise ValueError(
                    f'motif_coords must have shape ({seq_length}, 3, 3), '
                    f'got {motif_coords.shape}.')
        if device is None:
            device = self.device

        was_training = self.model.training
        try:
            if self.architecture == 'transformer':
                shape = (num_samples, seq_length, self.coord_dim)
                samples = self.schedule.sample(self.model, shape, device)
            else:
                samples = self._generate_multitrack(num_samples, seq_length,
                                                    device, motif_mask,
                                                    motif_coords)
        finally:
            self.model.train(was_training)

        result = samples.cpu().numpy()
        if self._train_std is not None and self._train_std > 1e-6:
            result = result * self._train_std
        if self._train_mean is not None:
            result = result + np.tile(self._train_mean, 3)
        return result

    def _generate_multitrack(
            self, num_samples: int, seq_length: int, device: torch.device,
            motif_mask: Optional[np.ndarray],
            motif_coords: Optional[np.ndarray]) -> torch.Tensor:
        """Run the combined SO(3) + translation reverse diffusion loop.

        Parameters
        ----------
        num_samples : int
            Number of backbone structures to generate.
        seq_length : int
            Number of residues in each generated structure.
        device : torch.device
            Device to run sampling on.
        motif_mask : np.ndarray, optional
            Boolean array of shape ``(seq_length,)`` flagging fixed motif
            positions, shared across the whole batch.
        motif_coords : np.ndarray, optional
            Reference ``(N, CA, C)`` coordinates of shape
            ``(seq_length, 3, 3)``.

        Returns
        -------
        torch.Tensor
            Generated ``(N, CA, C)`` coordinates flattened to shape
            ``(num_samples, seq_length, 9)``.
        """
        assert self.igso3 is not None
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

            fixed = None
            motif_rotations = motif_translations = None
            if motif_mask is not None:
                assert motif_coords is not None
                fixed = torch.tensor(motif_mask,
                                     dtype=torch.bool,
                                     device=device).expand(
                                         num_samples, seq_length)
                ref = torch.tensor(motif_coords,
                                   dtype=torch.float32,
                                   device=device)
                m_r, m_t = build_backbone_frames(ref)
                motif_rotations = m_r.expand(num_samples, seq_length, 3, 3)
                motif_translations = m_t.expand(num_samples, seq_length, 3)
                rotations = torch.where(fixed[..., None, None], motif_rotations,
                                        rotations)
                translations = torch.where(fixed[..., None], motif_translations,
                                           translations)

            for step in reversed(range(self.num_diffusion_steps)):
                t_batch = torch.full((num_samples,),
                                     step,
                                     dtype=torch.long,
                                     device=device)
                noisy_coords = backbone_coords_from_frames(
                    rotations, translations).reshape(num_samples, seq_length, 9)
                pred_rotations, pred_translations = self.model(
                    [noisy_coords, rotations, translations, t_batch, mask])
                if step > 0:
                    rotations = so3_x0_reverse_step(self.igso3, rotations,
                                                    pred_rotations, step)
                    translations = translation_posterior_step(
                        self.schedule, pred_translations, translations, step)
                else:
                    rotations, translations = pred_rotations, pred_translations

                if fixed is not None:
                    assert (motif_rotations is not None and
                            motif_translations is not None)
                    rotations = torch.where(fixed[..., None, None],
                                            motif_rotations, rotations)
                    translations = torch.where(fixed[..., None],
                                               motif_translations, translations)

            return backbone_coords_from_frames(rotations, translations).reshape(
                num_samples, seq_length, 9)

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir: Optional[str] = None) -> None:
        """Save model weights and training normalization statistics.

        Calls the parent ``TorchModel.save_checkpoint`` and then appends the
        running CA-centroid (``_train_mean``) and coordinate std
        (``_train_std``) to the checkpoint file. These are needed so that
        ``restore`` can produce correctly denormalized samples after reloading.

        Parameters
        ----------
        max_checkpoints_to_keep : int, default 5
            Maximum number of recent checkpoints to keep on disk. Older ones
            are deleted automatically.
        model_dir : str, optional
            Directory to save checkpoints in. Defaults to the model's
            ``model_dir``.

        Returns
        -------
        None
        """
        super().save_checkpoint(max_checkpoints_to_keep=max_checkpoints_to_keep,
                                model_dir=model_dir)
        if max_checkpoints_to_keep == 0:
            return
        checkpoint = sorted(self.get_checkpoints(model_dir))[0]
        data = torch.load(checkpoint, map_location=self.device)
        data['rf_diffusion_train_mean'] = (None if self._train_mean is None else
                                           self._train_mean.tolist())
        data['rf_diffusion_train_std'] = self._train_std
        torch.save(data, checkpoint)

    def restore(self,
                checkpoint: Optional[str] = None,
                model_dir: Optional[str] = None,
                strict: Optional[bool] = True) -> None:
        """Restore model weights and training normalization statistics.

        Calls the parent ``TorchModel.restore`` and then reads
        ``rf_diffusion_train_mean`` and ``rf_diffusion_train_std`` from the
        checkpoint file (written by ``save_checkpoint``). If those keys are
        absent the normalization statistics remain ``None``.

        Parameters
        ----------
        checkpoint : str, optional
            Path to a specific checkpoint file. If ``None``, the most recent
            checkpoint in ``model_dir`` is used.
        model_dir : str, optional
            Directory containing checkpoints. Ignored when ``checkpoint`` is
            provided directly.
        strict : bool, optional, default True
            Passed to ``torch.nn.Module.load_state_dict``. Set to ``False``
            to allow loading a checkpoint with missing or extra keys.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no checkpoint is found in ``model_dir`` and ``checkpoint`` is
            ``None``.
        """
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if not checkpoints:
                raise ValueError('No checkpoint found.')
            checkpoint = checkpoints[0]
        super().restore(checkpoint=checkpoint,
                        model_dir=model_dir,
                        strict=strict)
        data = torch.load(checkpoint, map_location=self.device)
        train_mean = data.get('rf_diffusion_train_mean')
        self._train_mean = (None if train_mean is None else np.asarray(
            train_mean, dtype=np.float32))
        self._train_std = data.get('rf_diffusion_train_std')
