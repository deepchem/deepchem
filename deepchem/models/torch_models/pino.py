"""
PINO: Physics-Informed Neural Operator

This module implements a PINO model for solving PDEs (e.g. Burgers' equation) using Fourier
based neural operators combined with physics-informed loss terms. The model inherits from TorchModel
and integrates additional loss terms for physics residuals and boundary enforcement.

References:
    https://arxiv.org/abs/2111.03794
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Callable, Dict, Tuple
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# --------------------- Model Components ---------------------
class ActNorm(nn.Module):
    """
    Activation normalization layer that applies learnable scaling and bias.

    The layer applies the transformation:
        y = scale * x + bias

    Parameters
    ----------
    num_features : int
        Number of features/channels in the input.

    Examples
    --------
    >>> norm = ActNorm(dim=64)
    >>> x = torch.randn(10, 1, 64)
    >>> y = norm(x)
    >>> print(y.shape)
    torch.Size([10, 1, 64])
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.scale + self.bias
        return out


class FourierBlock1D(nn.Module):
    """
    Single 1D Fourier block that performs a spectral convolution followed by a linear transformation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    modes : int
        Number of Fourier modes to retain in the frequency domain.

    Examples
    --------
    >>> block = FourierBlock1D(in_channels=64, out_channels=64, modes=16)
    >>> x = torch.randn(10, 64, 128)
    >>> y = block(x)
    >>> print(y.shape)
    torch.Size([10, 128, 64])
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)**0.5
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 2))
        self.norm = ActNorm(out_channels)
        logger.debug(
            f"[FourierBlock1D] Initialized with weights shape: {self.weights.shape}"
        )

    def compl_mul1d(self, a, b):
        real = torch.einsum("bci,oci->boi",
                            a[..., 0], b[..., 0]) - torch.einsum(
                                "bci,oci->boi", a[..., 1], b[..., 1])
        imag = torch.einsum("bci,oci->boi",
                            a[..., 0], b[..., 1]) + torch.einsum(
                                "bci,oci->boi", a[..., 1], b[..., 0])
        result = torch.stack([real, imag], dim=-1)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        x_ft = torch.view_as_real(x_ft)
        out_ft = torch.zeros(B,
                             self.out_channels,
                             x_ft.shape[-2],
                             2,
                             device=x.device)
        limit = min(self.modes, x_ft.shape[-2])
        out_ft[:, :, :limit] = self.compl_mul1d(x_ft[:, :, :limit],
                                                self.weights[:, :, :limit])
        x_rec = torch.fft.irfft(torch.view_as_complex(out_ft), n=L, dim=-1)
        x_rec = x_rec.permute(0, 2, 1)
        x_rec = self.norm(x_rec)
        return x_rec


# --------------------- PINO Model ---------------------
class PINO(TorchModel):
    """
    Physics-Informed Neural Operator (PINO) model for solving PDEs.

    This class implements a 1D Fourier Neural Operator augmented with physics-informed loss
    terms to solve partial differential equations such as Burgers' equation. The model integrates
    data, boundary, and physics losses and supports learnable PDE parameters.

    Parameters
    ----------
    in_channels: int, default 1
        Number of input channels (e.g., 2 for (x, t) coordinates).
    param_dim: int, default 0
        Dimension of the learnable PDE parameter input.
    modes: int, default 16
        Number of Fourier modes to retain.
    width: int, default 64
        Width of the hidden layers in the Fourier network.
    pde_fn: Callable
        A function that defines the PDE residual. Must accept (inputs, outputs, phys_params).
    boundary_data: Dict, default {}
        Dictionary containing boundary condition data used for computing boundary loss.
    boundary_weight: float, default 50.0
        Weight multiplier for the boundary loss term.
    data_weight: float, default 1.0
        Weight multiplier for the supervised data loss term.
    physics_weight: float, default 100
        Weight multiplier for the physics loss (PDE residual).
    hi_res: Tuple[int], default (256,)
        High-resolution grid shape used during training to estimate physics loss.
    **kwargs:
        Additional keyword arguments passed to the TorchModel base class.

    Examples
    --------
    (note that `dataset` here is only illustrative)
    >>> import torch
    >>> import numpy as np
    >>> from deepchem.models.torch_models.pino import PINO
    >>> # Define a dummy PDE function for testing:
    >>> def burgers_pde(inputs, outputs, phys_params): return outputs - inputs  # dummy residual function
    >>> # Create a dummy PINO model.
    >>> model = PINO(in_channels=2, modes=16, width=64, pde_fn=burgers_pde, phys_params={"alpha": torch.tensor(np.log(0.01))})
    >>> # The following line is commented out as it requires a valid dataset.
    >>> # model.fit(dataset, nb_epoch=100)
    >>> model is not None
    True
    """

    def __init__(
            self,
            in_channels: int = 1,
            param_dim: int = 0,
            modes: int = 16,
            width: int = 64,
            pde_fn: Callable = None,
            boundary_data: Dict = {},
            boundary_weight: float = 50.0,
            data_weight: float = 1.0,
            physics_weight: float = 100,
            hi_res: Tuple[int] = (256,),
            **kwargs,
    ):
        if pde_fn is None:
            raise ValueError("PDE residual function must be provided")
        self.boundary_weight = kwargs.pop("boundary_weight", 1.0)
        self.phys_params = nn.ParameterDict()
        if "phys_params" in kwargs:
            for name, value in kwargs["phys_params"].items():
                self.phys_params[name] = nn.Parameter(value)
        self.pde_fn = pde_fn
        self.boundary_data = boundary_data
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
        self.hi_res = hi_res
        self.data_loss_fn = nn.MSELoss()
        self._inputs = None
        self._hi_res_inputs = None
        self._is_training = False
        model = self.FNO1D(in_channels, 0, modes, width)
        super().__init__(model=model, loss=self._loss_fn, **kwargs)

    def parameters(self):
        all_params = list(self.model.parameters()) + list(
            self.phys_params.values())
        logger.debug(f"[PINO] Total parameters count: {len(all_params)}")
        return all_params

    class FNO1D(nn.Module):
        """
        1D Fourier Neural Operator model consisting of a stack of Fourier layers.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        modes : int
            Number of Fourier modes to retain.
        width : int
            Width of the hidden representation.

        Examples
        --------
        >>> from deepchem.models.torch_models.pino import PINO
        >>> model = PINO.FNO1D(in_channels=2, param_dim=0, modes=16, width=64)
        >>> x = torch.randn(10, 128, 2)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([10, 128, 1])
        """

        def __init__(self, in_channels, param_dim, modes, width):
            super().__init__()
            self.param_encoder = (nn.Linear(param_dim, width //
                                            2) if param_dim > 0 else None)
            fc0_in_dim = in_channels + (width // 2 if self.param_encoder
                                        is not None else 0)
            self.fc0 = nn.Linear(fc0_in_dim, width)
            self.fourier_blocks = nn.ModuleList(
                [FourierBlock1D(width, width, modes) for _ in range(4)])
            self.fc1 = nn.Linear(width, 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self,
                    x: torch.Tensor,
                    params: torch.Tensor = None) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if self.param_encoder is not None:
                if params is None:
                    params = torch.zeros(x.shape[0],
                                         self.param_encoder.in_features,
                                         device=x.device)
                p_emb = self.param_encoder(params)
                p_emb_exp = p_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x, p_emb_exp], dim=-1)
            x = self.fc0(x)
            z = x.permute(0, 2, 1)
            for i, block in enumerate(self.fourier_blocks):
                out = block(z)
                out = out.permute(0, 2, 1)
                z = z + out
            x = z.permute(0, 2, 1)
            x = F.gelu(self.fc1(x))
            output = self.fc2(x)
            return output

    def _create_hi_res_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        B, L, D = inputs.shape
        new_L = self.hi_res[0]
        new_grid = torch.linspace(0, 1, new_L, device=inputs.device)
        hi_res_inputs = new_grid.unsqueeze(0).repeat(B, 1).unsqueeze(-1)
        logger.debug(
            f"[PINO] Created hi_res_inputs with shape: {hi_res_inputs.shape}")
        return hi_res_inputs

    def _loss_fn(self, outputs, labels, weights):
        if isinstance(outputs, list):
            if len(outputs) == 1:
                model_output = outputs[0]
            else:
                raise TypeError(
                    "Expected outputs to be a single tensor or list with one tensor."
                )
        elif isinstance(outputs, torch.Tensor):
            model_output = outputs
        else:
            raise TypeError(f"Unexpected outputs type: {type(outputs)}")
        device = self._inputs.device if self._inputs is not None else torch.device(
            "cpu")
        if not isinstance(labels, torch.Tensor):
            labels_tensor = torch.as_tensor(np.array(labels),
                                            dtype=torch.float32,
                                            device=device)
            logger.debug("[LossFn] Converted labels to tensor.")
        else:
            labels_tensor = labels.to(device)
        if labels_tensor.ndim == 2:
            labels_tensor = labels_tensor.unsqueeze(-1)
        data_loss = self.data_loss_fn(model_output, labels_tensor)
        u_for_pde = self.model(self._inputs)
        physics_loss = self._compute_pde_loss(self._inputs, u_for_pde)
        boundary_loss = self._compute_boundary_loss()
        total_loss = (self.data_weight * data_loss +
                      self.physics_weight * physics_loss +
                      self.boundary_weight * boundary_loss)
        return total_loss

    def fit(self, dataset, nb_epoch=10, **kwargs):
        """
        Trains the PINO model using data, physics, and boundary losses.

        Parameters
        ----------
        dataset : deepchem.data.Dataset
            Dataset containing input tensors and target tensors.
        nb_epoch : int, default 10
            Number of training epochs.
        **kwargs:
            Additional keyword arguments. Supports:
            - phys_lr: float, default 1e-3
                Learning rate for physical parameters (e.g., viscosity `nu`).

        Returns
        -------
        List[float]
            List containing average loss for each epoch.
        """
        self._is_training = True
        loss_history = []
        phys_lr = kwargs.get("phys_lr", 1e-3)
        logger.debug(
            f"[PINO] Starting training with custom optimizer. Using model LR=1e-4 and phys_lr={phys_lr}"
        )
        optimizer = torch.optim.Adam([
            {
                "params": self.model.parameters(),
                "lr": 1e-4
            },
            {
                "params": self.phys_params.values(),
                "lr": phys_lr
            },
        ])
        grad_clip_threshold = 0.5
        device = self.parameters()[0].device  # noqa: F841 Retain for future use
        for epoch in range(nb_epoch):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            for batch in dataset.iterbatches():
                inputs, labels, weights = self._prepare_batch(batch)
                optimizer.zero_grad()
                outputs = self.model(inputs, params=None)
                loss = self._loss_fn(outputs, labels, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(),
                                               grad_clip_threshold)
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            epoch_loss /= (batch_count if batch_count > 0 else 1)
            loss_history.append(epoch_loss)
            current_nu = torch.exp(self.phys_params["alpha"]).item()
            logger.debug(
                f"[Training] Epoch {epoch + 1}/{nb_epoch}, Loss: {epoch_loss:.4f}, nu: {current_nu:.6f}"
            )
        self._is_training = False
        logger.debug("[PINO] Training complete.")
        return loss_history

    def _prepare_batch(self, batch):
        """
        Converts a batch from DeepChem Dataset to torch tensors.

        Parameters
        ----------
        batch : Tuple[np.ndarray, np.ndarray]
            A tuple of (X, y), where:
            - X is the input array of shape (batch, spatial, features)
            - y is the output array of shape (batch, spatial, 1)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing input and output tensors suitable for the model.
        """
        if isinstance(batch, list) and isinstance(batch[0], dict):
            inputs = torch.stack([d["input"] for d in batch], dim=0)
            labels = torch.stack([d["label"] for d in batch], dim=0)
            weights = None
        elif isinstance(batch, tuple):
            logger.debug(
                f"[_prepare_batch] Batch provided as tuple of length {len(batch)}."
            )
            inputs = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            weights = batch[2] if len(batch) > 2 else None
        else:
            inputs = batch
            labels = None
            weights = None
        logger.debug("DEBUG: type(inputs): %s", type(inputs))
        try:
            logger.debug("DEBUG: np.shape(inputs): %s", np.shape(inputs))
        except Exception as e:
            logger.debug("DEBUG: np.shape failed: %s", e)
        if isinstance(inputs, np.ndarray):
            logger.debug("DEBUG: Detected inputs as numpy array; shape: %s",
                         inputs.shape)
            inputs = torch.from_numpy(inputs).float()
        elif not isinstance(inputs, torch.Tensor):
            try:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                logger.debug("DEBUG: Converted inputs to tensor; shape: %s",
                             inputs.shape)
            except Exception as e:
                logger.error("Error converting inputs to torch tensor: %s", e)
                raise
        if inputs.ndim == 2:
            logger.debug(
                "DEBUG: inputs has 2 dimensions; unsqueezing to get (B, 1, D)")
            inputs = inputs.unsqueeze(1)
            logger.debug("DEBUG: New inputs shape: %s", inputs.shape)
        if labels is not None and not isinstance(labels, torch.Tensor):
            if isinstance(labels, np.ndarray):
                logger.debug("DEBUG: Detected labels as numpy array; shape: %s",
                             labels.shape)
                labels = torch.from_numpy(labels).float()
            else:
                try:
                    labels = torch.tensor(labels, dtype=torch.float32)
                    logger.debug("DEBUG: Converted labels to tensor; shape: %s",
                                 labels.shape)
                except Exception as e:
                    logger.error("Error converting labels to torch tensor: %s",
                                 e)
                    raise
        self._inputs = inputs.clone().detach().requires_grad_(True)
        if self._is_training and self.hi_res:
            self._hi_res_inputs = self._create_hi_res_inputs(inputs)
        else:
            self._hi_res_inputs = torch.randn(32,
                                              self.hi_res[0],
                                              2,
                                              device=inputs.device)
        return inputs, labels, weights

    def _compute_pde_loss(self, inputs: torch.Tensor,
                          _unused: torch.Tensor) -> torch.Tensor:
        """
        Computes the physics-informed loss from the PDE residual given the input data.

        This method evaluates the residual for the PDE of the form:
            u_t + u * u_x - ν * u_xx = 0,
        where:
        - u is the model output computed from the inputs.
        - u_t and u_x are the temporal and spatial gradients of u (computed via autograd).
        - u_xx is the second spatial derivative.
        - ν (nu) is obtained as the exponential of a learnable parameter `alpha`.

        The loss is defined as the mean squared error of the residual scaled by 1e4.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, spatial, features). This tensor will be set to require gradients
            if it does not already.
        _unused : torch.Tensor
            Placeholder parameter for interface compatibility. Its value is not used in the loss calculation.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the mean squared PDE residual loss.
        """

        logger.debug("[_compute_pde_loss] Computing PDE loss...")
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        u = self.model(inputs)
        grads = torch.autograd.grad(
            outputs=u,
            inputs=inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        if grads is None:
            logger.debug(
                "[_compute_pde_loss] Warning: grads is None, returning PDE loss = 0"
            )
            return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
        logger.debug(f"[_compute_pde_loss] grad shape: {grads.shape}")
        u_x = grads[..., 0:1]
        u_t = grads[..., 1:2]
        logger.debug(
            f"[_compute_pde_loss] u_x shape: {u_x.shape}, u_t shape: {u_t.shape}"
        )
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=inputs,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][..., 0:1]
        logger.debug(f"[_compute_pde_loss] u_xx shape: {u_xx.shape}")
        alpha = self.phys_params[
            "alpha"] if "alpha" in self.phys_params else torch.tensor(
                np.log(0.01), device=inputs.device, requires_grad=True)
        nu = torch.exp(alpha)
        logger.debug(
            f"[_compute_pde_loss] Using nu = exp(alpha) value: {nu.item()}")
        residual = u_t + u * u_x - nu * u_xx
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            logger.debug(
                "[_compute_pde_loss] Warning: NaN or Inf in PDE residual!")
            return torch.tensor(1e6, device=inputs.device, dtype=inputs.dtype)
        pde_loss = torch.mean(residual**2) * 10000
        logger.debug(f"[_compute_pde_loss] PDE loss: {pde_loss.item()}")
        logger.debug(
            f"[_compute_pde_loss] u range: {u.min().item():.4f} to {u.max().item():.4f}"
        )
        logger.debug(
            f"[_compute_pde_loss] u_t range: {u_t.min().item():.4f} to {u_t.max().item():.4f}"
        )
        logger.debug(
            f"[_compute_pde_loss] u_x range: {u_x.min().item():.4f} to {u_x.max().item():.4f}"
        )
        logger.debug(
            f"[_compute_pde_loss] u_xx range: {u_xx.min().item():.4f} to {u_xx.max().item():.4f}"
        )
        logger.debug(
            f"[_compute_pde_loss] residual range: {residual.min().item():.4f} to {residual.max().item():.4f}"
        )
        logger.debug(f"[_compute_pde_loss] nu value (exp(alpha)): {nu.item()}")
        logger.debug(
            f"[_compute_pde_loss] Calculated PDE loss: {pde_loss.item()}")
        return pde_loss

    def _compute_boundary_loss(self) -> torch.Tensor:
        """
        Computes the loss associated with enforcing boundary conditions.

        For each boundary condition specification in `self.boundary_data`, if both `points`
        and `values` are provided, the method computes the model prediction at the given
        boundary points and then calculates the mean squared error between these predictions
        and the specified boundary values. The losses from individual boundary sets are summed
        to obtain the final boundary loss.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the boundary loss. If no boundary data is provided,
            a zero loss is returned.
        """
        logger.debug("[_compute_boundary_loss] Computing boundary loss...")
        if not self.boundary_data:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        device = next(self.model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for key, value in self.boundary_data.items():
            if isinstance(value, dict):
                points = value.get("points")
                values = value.get("values")
                if points is not None and values is not None:
                    logger.debug(
                        f"[_compute_boundary_loss] Processing boundary '{key}' with points shape: {points.shape}"
                    )
                    pred = self.model(points)
                    if key.lower() == "dirichlet":
                        loss_val = torch.mean((pred - values)**2)
                        logger.debug(
                            f"[_compute_boundary_loss] Boundary loss for '{key}': {loss_val.item()}"
                        )
                        loss += loss_val
        return loss

    def _compute_physics_loss(self, inputs: torch.Tensor,
                              u: torch.Tensor) -> torch.Tensor:
        return self._compute_pde_loss(inputs, u)

    def predict(self, dataset, params: np.ndarray = None):
        logger.debug("[predict] Starting prediction...")
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataset.iterbatches():
                inputs, _, _ = self._prepare_batch(batch)
                if params is not None:
                    params_tensor = torch.as_tensor(params,
                                                    dtype=torch.float32,
                                                    device=inputs.device)
                    logger.debug(
                        f"[predict] Using provided params with shape: {params_tensor.shape}"
                    )
                else:
                    params_tensor = None
                    logger.debug("[predict] No params provided.")
                output = self.model(inputs, params_tensor)
                logger.debug(f"[predict] Output shape: {output.shape}")
                preds.append(output)
        final_preds = torch.cat(preds, dim=0)
        logger.debug(f"[predict] Final predictions shape: {final_preds.shape}")
        return final_preds
