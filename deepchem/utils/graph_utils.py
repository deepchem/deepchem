import numpy as np


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """
    Fourier encode the input array `x` based on the specified number of encodings.

    This function applies a Fourier encoding to the input array `x` by dividing
    it by a range of scales (2^i for i in range(num_encodings)) and then
    concatenating the sine and cosine of the scaled values. Optionally, the
    original input array can be included in the output.

    Parameters
    ----------
    x : np.ndarray
        Input array to be Fourier encoded.
    num_encodings : int, optional, default=4
        Number of Fourier encodings to apply.
    include_self : bool, optional, default=True
        Whether to include the original input array in the output.

    Returns
    -------
    np.ndarray
        Fourier encoded array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> encoded_x = fourier_encode_dist_np(x, num_encodings=4, include_self=True)
    """
    x = x[..., np.newaxis]
    dtype, orig_x = x.dtype, x
    scales = 2**np.arange(num_encodings, dtype=dtype)
    x = x / scales
    x = np.concatenate([np.sin(x), np.cos(x)], axis=-1)
    x = np.concatenate((x, orig_x), axis=-1) if include_self else x
    return x.squeeze()


EPS = 1e-5


def aggregate_mean(h, **kwargs):
    """
    Compute the mean of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.

    Returns
    -------
    np.array
        Mean of the input tensor along the second to last dimension.
    """
    return np.mean(h, axis=-2)


def aggregate_max(h, **kwargs):
    """
    Compute the max of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.

    Returns
    -------
    np.array
        Max of the input tensor along the second to last dimension.
    """
    return np.max(h, axis=-2)


def aggregate_min(h, **kwargs):
    """
    Compute the min of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    np.array
        Min of the input tensor along the second to last dimension.
    """
    return np.min(h, axis=-2)


def aggregate_std(h, **kwargs):
    """
    Compute the standard deviation of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.

    Returns
    -------
    np.array
        Standard deviation of the input tensor along the second to last dimension.
    """
    return np.sqrt(aggregate_var(h) + 1e-5)


def aggregate_var(h, **kwargs):
    """
    Compute the variance of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.

    Returns
    -------
    np.array
        Variance of the input tensor along the second to last dimension.
    """
    h_mean_squares = np.mean(h * h, axis=-2)
    h_mean = np.mean(h, axis=-2)
    var = np.maximum(h_mean_squares - h_mean * h_mean, 0)
    return var


def aggregate_moment(h, n=3, **kwargs):
    """
    Compute the nth moment of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.
    n : int, optional, default=3
        The order of the moment to compute.

    Returns
    -------
    np.array
        Nth moment of the input tensor along the second to last dimension.
    """
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = np.mean(h, axis=-2, keepdims=True)
    h_n = np.mean(np.power(h - h_mean, n), axis=-2)
    rooted_h_n = np.sign(h_n) * np.power(np.abs(h_n) + 1e-5, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    """
    Compute the sum of the input tensor along the second to last dimension.

    Parameters
    ----------
    h : np.array
        Input tensor.

    Returns
    -------
    np.array
        Sum of the input tensor along the second to last dimension.
    """
    return np.sum(h, axis=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output
def scale_identity(h, D=None, avg_d=None):
    """
    Identity scaling function.

    Parameters
    ----------
    h : np.array
        Input tensor.
    D : np.array, optional
        Degree tensor.
    avg_d : dict, optional
        Dictionary containing averages over the training set.

    Returns
    -------
    np.array
        Scaled input tensor.
    """
    return h


def scale_amplification(h, D, avg_d):
    """
    Amplification scaling function. log(D + 1) / d * h where d is the average of the ``log(D + 1)`` in the training set

    Parameters
    ----------
    h : np.array
        Input tensor.
    D : np.array
        Degree tensor.
    avg_d : dict
        Dictionary containing averages over the training set.

    Returns
    -------
    np.array
        Scaled input tensor.
    """
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    """
    Attenuation scaling function. (log(D + 1))^-1 / d * X where d is the average of the ``log(D + 1))^-1`` in the training set

    Parameters
    ----------
    h : np.array
        Input tensor.
    D : np.array
        Degree tensor.
    avg_d : dict
        Dictionary containing averages over the training set.

    Returns
    -------
    np.array
        Scaled input tensor.
    """
    return h * (avg_d["log"] / np.log(D + 1))
