import torch
from typing import Optional, List


class GaussianNormalizer:
    """
    The GaussianNormalizer transforms input data by subtracting the mean and dividing by the standard deviation,
    resulting in normalized data with zero mean and unit variance. This is also known as standardization or z-score normalization.

    The transformation is:
        x_norm = (x - mean) / std

    And the inverse transformation is:
        x = x_norm * std + mean

    The normalizer can compute statistics over specified dimensions (e.g. batch and spatial dims)
    while preserving others (e.g. channels), which is useful for multi-dimensional data like images
    or solutions to PDEs.

    Example
    -------------
    >>> import torch
    >>> from deepchem.utils.fno_utils import GaussianNormalizer
    >>> normalizer = GaussianNormalizer()
    >>> data = torch.randn(100, 10)
    >>> normalizer.fit(data)
    >>> normalized_data = normalizer.transform(data)
    >>> denormalized_data = normalizer.inverse_transform(normalized_data)
    """

    def __init__(self,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 eps: float = 1e-7,
                 dim: Optional[List[int]] = None):
        """
        Parameters
        ----------
        mean: torch.Tensor, optional
            Pre-computed mean tensor
        std: torch.Tensor, optional
            Pre-computed standard deviation tensor
        eps: float, default 1e-7
            Small epsilon for numerical stability
        dim: List[int], optional
            Dimensions to reduce over for computing mean/std
            If None, reduces over all dimensions
        """
        self.mean = mean
        self.std = std
        self.eps = eps
        self.dim = dim
        self.fitted = False

    def fit(self, data: torch.Tensor) -> 'GaussianNormalizer':
        """Fit the normalizer to training data.

        Parameters
        ----------
        data: torch.Tensor
            Training data to compute statistics from

        Returns
        -------
        self
        """
        if self.dim is None:
            self.mean = torch.mean(data)
            self.std = torch.std(data)
        else:
            self.mean = torch.mean(data, dim=self.dim, keepdim=True)
            self.std = torch.std(data, dim=self.dim, keepdim=True)
        self.fitted = True
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data using fitted statistics.

        Parameters
        ----------
        data: torch.Tensor
            Data to normalize

        Returns
        -------
        torch.Tensor
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        if self.std is None or self.mean is None:
            return data
        else:
            return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale.

        Parameters
        ----------
        data: torch.Tensor
            Normalized data to denormalize

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        if not self.fitted:
            raise ValueError(
                "Normalizer must be fitted before inverse_transform")
        if self.std is None or self.mean is None:
            return data
        else:
            return data * (self.std + self.eps) + self.mean

    def to(self, device: torch.device) -> 'GaussianNormalizer':
        """Move normalizer to device."""
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self
