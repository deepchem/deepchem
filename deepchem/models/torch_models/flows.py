""" This module contains the implementation of the various flow layers and models"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class Flow(nn.Module):
    """
    Generic class for flow functions

    Flows [flow1]_ should satisfy several conditions in order to be practical. They should:

    - be invertible; for sampling we need g while for computing likelihood we need `f` ,
    - be sufficiently expressive to model the distribution of interest,
    - be computationally efficient, both in terms of computing `f` and `g` (depending on the application) but also in terms of the calculation of the determinant of the Jacobian.

    Flow layers are generally used as a part of a Normalizing Flow model,
    which is a generative model that learns a target distribution by transforming a
    simple base distribution through a series of invertible transformations.
    The target distribution is then defined as the composition of the base distribution
    and the flow transformations.

    References
    ----------
    .. [flow1] Kobyzev, I., Prince, S. J., & Brubaker, M. A. (2020).
        Normalizing flows: An introduction and review of current methods.
        IEEE transactions on pattern analysis and machine intelligence, 43(11), 3964-3979.
    """

    def __init__(self):
        """Initializes the flow function
        """
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the flow

        Parameters
        ----------
        z: torch.Tensor
            Input tensor

        Returns
        -------
        z_: torch.Tensor
            Transformed tensor
        log_det: torch.Tensor
            Logarithm of the determinant of the Jacobian of the transformation
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass of the flow

        Parameters
        ----------
        z: torch.Tensor
            Input tensor

        Returns
        -------
        z_: torch.Tensor
            Transformed tensor
        log_det: torch.Tensor
            Logarithm of the determinant of the Jacobian of the transformation
        """
        raise NotImplementedError("This flow has no algebraic inverse.")


class Affine(Flow):
    """Class which performs the Affine transformation.

    This transformation is based on the affinity of the base distribution with
    the target distribution. A geometric transformation is applied where
    the parameters performs changes on the scale and shift of a function
    (inputs).

    Normalizing Flow transformations must be bijective in order to compute
    the logarithm of jacobian's determinant. For this reason, transformations
    must perform a forward and inverse pass.

    Example
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.layers import Affine
    >>> import torch
    >>> from torch.distributions import MultivariateNormal
    >>> # initialize the transformation layer's parameters
    >>> dim = 2
    >>> samples = 96
    >>> transforms = Affine(dim)
    >>> # forward pass based on a given distribution
    >>> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> input = distribution.sample(torch.Size((samples, dim)))
    >>> len(transforms.forward(input))
    2
    >>> # inverse pass based on a distribution
    >>> len(transforms.inverse(input))
    2

    """

    def __init__(self, dim: int) -> None:
        """Create a Affine transform layer.

        Parameters
        ----------
        dim: int
            Value of the Nth dimension of the dataset.

        """

        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(self.dim))
        self.shift = nn.Parameter(torch.zeros(self.dim))
        self.batch_dims = torch.nonzero(torch.tensor(self.scale.shape) == 1,
                                        as_tuple=False)[:, 0].tolist()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a transformation between two different distributions. This
        particular transformation represents the following function:

        .. math:: y = x * exp(a) + b

        where a is scale parameter and b performs a shift.
        This class also returns the logarithm of the jacobians determinant
        which is useful when invert a transformation and compute the
        probability of the transformation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor sample with the initial distribution data which will pass into
            the normalizing flow algorithm.

        Returns
        -------
        y : torch.Tensor
            Transformed tensor according to Affine layer with the shape of 'x'.
        log_det_jacobian : torch.Tensor
            Tensor which represents the info about the deviation of the initial
            and target distribution.

        """

        y = torch.exp(self.scale) * x + self.shift
        det_jacobian = torch.exp(self.scale.sum())
        log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

        return y, log_det_jacobian

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a transformation between two different distributions.
        This transformation represents the bacward pass of the function
        mention before. Its mathematical representation is x = (y - b) / exp(a)
        , where "a" is scale parameter and "b" performs a shift. This class
        also returns the logarithm of the jacobians determinant which is
        useful when invert a transformation and compute the probability of
        the transformation.

        Parameters
        ----------
        y : torch.Tensor
            Tensor sample with transformed distribution data which will be used in
            the normalizing algorithm inverse pass.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor according to Affine layer with the shape of 'y'.
        inverse_log_det_jacobian : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.

        """

        x = (y - self.shift) / torch.exp(self.scale)
        det_jacobian = 1 / torch.exp(self.scale.sum())
        inverse_log_det_jacobian = torch.ones(
            x.shape[0]) * torch.log(det_jacobian)

        return x, inverse_log_det_jacobian


class MaskedAffineFlow(Flow):
    """
    This class implements the Masked Affine Flow layer

    The Masked Affine Flow [maskedaffine1]_ layer is a type of normalizing flow layer which
    is used to learn a target distribution. The layer is based on the
    affine flow layer, but with a mask applied to the input data. The mask
    is a tensor of the same size as the input data, filled with 0s and 1s.
    The mask is used to determine which features are transformed by the
    affine flow layer. The affine flow layer is defined as follows:

    Masked affine flow
    .. math:: f(z) = b * z + (1 - b) * (z * e^{s(b * z)} + t)

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> from deepchem.models.torch_models.flows import MaskedAffineFlow
    >>> from torch.distributions import MultivariateNormal

    >>> dim = 2
    >>> samples = 96
    >>> data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> tensor = data.sample(torch.Size((samples, dim)))

    >>> layers = 4
    >>> hidden_size = 16
    >>> masks = F.one_hot(torch.tensor([i % 2 for i in range(layers)])).float()

    >>> s_func = nn.Sequential(
    ...     nn.Linear(in_features=dim, out_features=hidden_size), nn.LeakyReLU(),
    ...     nn.Linear(in_features=hidden_size, out_features=hidden_size),
    ...     nn.LeakyReLU(), nn.Linear(in_features=hidden_size, out_features=dim))

    >>> t_func = nn.Sequential(
    ...     nn.Linear(in_features=dim, out_features=hidden_size), nn.LeakyReLU(),
    ...     nn.Linear(in_features=hidden_size, out_features=hidden_size),
    ...     nn.LeakyReLU(), nn.Linear(in_features=hidden_size, out_features=dim))

    >>> layers = nn.ModuleList(
    ...     [MaskedAffineFlow(mask, s_func, t_func) for mask in masks])

    >>> for layer in layers:
    ...   _, inverse_log_det_jacobian = layer.inverse(tensor)
    ...   inverse_log_det_jacobian = inverse_log_det_jacobian.detach().numpy()
    >>> len(inverse_log_det_jacobian)
    96

    References
    ----------
    .. [maskedaffine1] Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016).
        Density estimation using real nvp. arXiv preprint arXiv:1605.08803.
    """

    def __init__(
        self,
        b: torch.Tensor,
        t: Optional[Union[torch.nn.ModuleList, torch.nn.Sequential]] = None,
        s: Optional[Union[torch.nn.ModuleList, torch.nn.Sequential]] = None
    ) -> None:
        """
        Initializes the Masked Affine Flow layer

        Parameters
        ----------
        b: torch.Tensor
            mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        t: Optional[Union[torch.nn.ModuleList, torch.nn.Sequential]], optional
            translation mapping, i.e. neural network, where first input dimension is batch dim,
            if None no translation is applied
        s: Optional[Union[torch.nn.ModuleList, torch.nn.Sequential]], optional
            scale mapping, i.e. neural network, where first input dimension is batch dim,
            if None no scale is applied
        """
        super().__init__()
        # self.b = b
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Masked Affine Flow layer

        Parameters
        ----------
        z : torch.Tensor
            Input tensor

        Returns
        -------
        z : torch.Tensor
            Transformed tensor according to Masked Affine Flow layer with the shape of 'z'.
        log_det : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.
        """
        z_masked: torch.Tensor = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale,
                            dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass of the Masked Affine Flow layer

        Parameters
        ----------
        z : torch.Tensor
            Input tensor

        Returns
        -------
        z_ : torch.Tensor
            Transformed tensor according to Masked Affine Flow layer with the shape of 'z'.
        log_det : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.
        """
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum(
            (1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


class ActNorm(Affine):
    """
    This class implements the ActNorm layer (for activation normalizaton)

    ActNorm is an Affine layer but with a data-dependent initialization,
    where on the very first batch we clever initialize the scale,shift so that the output
    is unit gaussian. As described in [glow]_ Kingma et al (2018).

    ActNorm is a layer that performs an affine transformation of the activations using a scale
    and bias parameter per channel, similar to batch normalization.
    These parameters are initialized such that the post-actnorm activations
    per-channel have zero mean and unit variance given an initial minibatch of data.
    This is a form of data dependent initialization [weight_norm]_.
    After initialization, the scale and bias are treated as regular trainable parameters
    that are independent of the data.

    Examples
    --------
    Importing necessary libraries

    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> from deepchem.models.torch_models.flows import MaskedAffineFlow
    >>> from torch.distributions import MultivariateNormal

    Creating sample data

    >>> dim = 2
    >>> samples = 96
    >>> data = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    >>> tensor = data.sample(torch.Size((samples, dim)))

    Initializing the ActNorm layer and performing forward and inverse pass

    >>> actnorm = ActNorm(dim)
    >>> _, log_det_jacobian = actnorm.forward(tensor)
    >>> _, inverse_log_det_jacobian = actnorm.inverse(tensor)
    >>> len(inverse_log_det_jacobian)
    96

    References
    ----------
    .. [glow] Kingma, D. P., & Dhariwal, P. (2018).
        Glow: Generative flow with invertible 1x1 convolutions.
        Advances in neural information processing systems, 31.
    .. [weight_norm] Salimans, T., & Kingma, D. P. (2016).
        Weight normalization: A simple reparameterization to
        accelerate training of deep neural networks.
        Advances in neural information processing systems, 29.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the ActNorm layer
        """
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = torch.tensor(0.0).detach().cpu()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the ActNorm layer

        Parameters
        ----------
        z : torch.Tensor
            Input tensor

        Returns
        -------
        z_ : torch.Tensor
            Transformed tensor according to ActNorm layer with the shape of 'z'.
        log_det : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.
        """
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:  # type: ignore
            assert self.scale is not None and self.shift is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.scale.data = s_init.data
            self.shift.data = (-z.mean(dim=self.batch_dims, keepdim=True) *
                               torch.exp(self.scale)).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass of the ActNorm layer

        Parameters
        ----------
        z : torch.Tensor
            Input tensor

        Returns
        -------
        z_ : torch.Tensor
            Transformed tensor according to ActNorm layer with the shape of 'z'.
        log_det : torch.Tensor
            Tensor which represents the information of the deviation of the initial
            and target distribution.
        """
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:  # type: ignore
            assert self.scale is not None and self.shift is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.scale.data = s_init.data
            self.shift.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)


class ClampExp(nn.Module):
    """
    A non Linearity layer that clamps the input tensor by taking the minimum of the
    exponential of the input multiplied by a lambda parameter and 1.

    .. math:: f(x) = min(exp(\lambda * x), 1)

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.flows import ClampExp
    >>> lambda_param = 1.0
    >>> clamp_exp = ClampExp(lambda_param)
    >>> input = torch.tensor([-1 ,0.5, 0.6, 0.7])
    >>> clamp_exp(input)
    tensor([0.3679, 1.0000, 1.0000, 1.0000])
    """

    def __init__(self, lambda_param: float = 1.0) -> None:
        """
        Initializes the ClampExp layer

        Parameters
        ----------
        lambda_param : float
            Lambda parameter for the ClampExp layer
        """

        self.lambda_param = lambda_param
        super(ClampExp, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ClampExp layer

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Transformed tensor according to ClampExp layer with the shape of 'x'.
        """
        one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(self.lambda_param * x), one)


class ConstScaleLayer(nn.Module):
    """
    This layer scales the input tensor by a fixed factor

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.flows import ConstScaleLayer
    >>> scale = 2.0
    >>> const_scale = ConstScaleLayer(scale)
    >>> input = torch.tensor([1, 2, 3])
    >>> const_scale(input)
    tensor([2., 4., 6.])
    """

    def __init__(self, scale: float = 1.0):
        """
        Initializes the ConstScaleLayer

        Parameters
        ----------
        scale : float
            Scaling factor
        """
        super().__init__()
        self.scale = torch.tensor(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConstScaleLayer

        Parameters
        ----------
        input : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Scaled tensor
        """
        return input * self.scale


class MLP_flow(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model for normalizing flows that is
    used as a part of a Normalizing Flow model.
    It is a modified version of the MLP model from `deepchem/deepchem/models/torch_models/layers.py`
    to handle multiple layers

    Example
    -------
    >>> import torch
    >>> from deepchem.models.torch_models.flows import MLP_flow
    >>> layers = [2, 4, 4, 2]
    >>> mlp_flow = MLP_flow(layers)
    >>> input = torch.tensor([1., 2.])
    >>> output = mlp_flow(input)
    >>> output.shape
    torch.Size([2])
    """

    def __init__(
        self,
        layers: list,
        leaky: float = 0.0,
        score_scale: Optional[float] = None,
        output_fn=None,
        output_scale: Optional[float] = None,
        init_zeros: bool = False,
        dropout: Optional[float] = None,
    ):
        """
        Initializes the MLP_flow model

        Parameters
        ----------
        layers : list
            List of layer sizes from start to end
        leaky : float, optional default 0.0
            Slope of the leaky part of the ReLU, if 0.0, standard ReLU is used
        score_scale : float, optional
            Factor to apply to the scores, i.e. output before output_fn
        output_fn : str, optional
            Function to be applied to the output, either None, "sigmoid", "relu", "tanh", or "clampexp"
        output_scale : float, optional
            Rescale outputs if output_fn is specified, i.e. scale * output_fn(out / scale)
        init_zeros : bool, optional
            Flag, if true, weights and biases of last layer are initialized with zeros
            (helpful for deep models, see arXiv 1807.03039)
        dropout : float, optional
            If specified, dropout is done before last layer; if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(ConstScaleLayer(score_scale))
            if output_fn == "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn == "relu":
                net.append(nn.ReLU())
            elif output_fn == "tanh":
                net.append(nn.Tanh())
            elif output_fn == "clampexp":
                net.append(ClampExp())
            else:
                NotImplementedError("This output function is not implemented.")
            if output_scale is not None:
                net.append(ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP_flow model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Transformed tensor according to the MLP_flow model with the shape of 'x'
        """
        return self.net(x)
