import numpy as np

try:
    import torch
except ImportError:
    print("This module requires PyTorch to be installed.")


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """
    Fourier encode the input tensor `x` based on the specified number of encodings.

    This function applies a Fourier encoding to the input tensor `x` by dividing
    it by a range of scales (2^i for i in range(num_encodings)) and then
    concatenating the sine and cosine of the scaled values. Optionally, the
    original input tensor can be included in the output.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be Fourier encoded.
    num_encodings : int, optional, default=4
        Number of Fourier encodings to apply.
    include_self : bool, optional, default=True
        Whether to include the original input tensor in the output.

    Returns
    -------
    torch.Tensor
        Fourier encoded tensor.

    Examples
    --------
    >>> import torch
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> encoded_x = fourier_encode_dist(x, num_encodings=4, include_self=True)
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2**torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze()


EPS = 1e-5


def aggregate_mean(h, **kwargs):
    """
    Compute the mean of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Mean of the input tensor along the second to last dimension.
    """
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    """
    Compute the max of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Max of the input tensor along the second to last dimension.
    """
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    """
    Compute the min of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    torch.Tensor
        Min of the input tensor along the second to last dimension.
    """
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    """
    Compute the standard deviation of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Standard deviation of the input tensor along the second to last dimension.
    """
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    """
    Compute the variance of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Variance of the input tensor along the second to last dimension.
    """
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    """
    Compute the nth moment of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    n : int, optional, default=3
        The order of the moment to compute.

    Returns
    -------
    torch.Tensor
        Nth moment of the input tensor along the second to last dimension.
    """
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    """
    Compute the sum of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Sum of the input tensor along the second to last dimension.
    """
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output
def scale_identity(h, D=None, avg_d=None):
    """
    Identity scaling function.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    D : torch.Tensor, optional
        Degree tensor.
    avg_d : dict, optional
        Dictionary containing averages over the training set.

    Returns
    -------
    torch.Tensor
        Scaled input tensor.
    """
    return h


def scale_amplification(h, D, avg_d):
    """
    Amplification scaling function. log(D + 1) / d * h where d is the average of the ``log(D + 1)`` in the training set

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    D : torch.Tensor
        Degree tensor.
    avg_d : dict
        Dictionary containing averages over the training set.

    Returns
    -------
    torch.Tensor
        Scaled input tensor.
    """
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    """
    Attenuation scaling function. (log(D + 1))^-1 / d * X where d is the average of the ``log(D + 1))^-1`` in the training set

    Parameters
    ----------
    h : torch.Tensor
        Input tensor.
    D : torch.Tensor
        Degree tensor.
    avg_d : dict
        Dictionary containing averages over the training set.

    Returns
    -------
    torch.Tensor
        Scaled input tensor.
    """
    return h * (avg_d["log"] / np.log(D + 1))
