import torch


def get_sinusoid(values, max_val, channels=32):
    """
    This function computes the sinusoidal embeddings for the values which are used to encode the position of the residues in the protein sequence.
    This is done by using the sinusoidal function and cosine function which are periodically
    alternated in the channels.

    This will be used to quantify positions in our 1D protein sequence embedding for RFdiffusion model.

    Parameters
    ----------
    values: torch.Tensor
        Values to get sinusoidal embeddings for.
    max_val: float
        Maximum value of the values.
    channels: int
        Number of channels.

    Returns
    -------
    torch.Tensor
        Sinusoidal embeddings for values.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import get_sinusoid
    >>> values = torch.tensor([1.0, 2.0, 3.0])
    >>> get_sinusoid(values, 10.0, 32)

    shape: [batch_size, num_values, channels]

    Returns
    -------
    torch.Tensor
        Sinusoidal embeddings for values.

    References
    ----------
    .. [1] "Attention is all you need", Vaswani, Ashish, et al.
           https://arxiv.org/abs/1706.03762

    """

    inv_freq = 1.0 / (10000**(
        torch.arange(0, channels, 2, dtype=torch.float32) / channels))
    scaled = values.unsqueeze(-1) * (1.0 / max_val)
    sin_f = torch.sin(scaled * inv_freq)
    cos_f = torch.cos(scaled * inv_freq)
    return torch.cat([sin_f, cos_f], dim=-1)
