import torch


def _take_eigpairs(eival: torch.Tensor, eivec: torch.Tensor, neig: int,
                   mode: str):
    """Take the eigenpairs from the eigendecomposition.

    Examples
    --------
    >>> import torch
    >>> eival = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    >>> eivec = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
    ...                       [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    >>> neig = 2
    >>> mode = "lowest"
    >>> eival, eivec = _take_eigpairs(eival, eivec, neig, mode)
    >>> eival
    tensor([[1., 2.],
            [4., 5.]])
    >>> eivec
    tensor([[[1., 2.],
             [4., 5.],
             [7., 8.]],
    <BLANKLINE>
            [[1., 2.],
             [4., 5.],
             [7., 8.]]])

    Parameters
    ----------
    eival: torch.Tensor
        Eigenvalues of the linear operator. Shape: ``(*BV, na)``.
    eivec: torch.Tensor
        Eigenvectors of the linear operator. Shape: ``(*BV, na, na)``.
    neig: int
        Number of eigenvalues and eigenvectors to be calculated.
    mode: str
        Mode of the eigenvalues to be calculated (``"lowest"``, ``"uppest"``)

    Returns
    -------
    eival: torch.Tensor
        Eigenvalues of the linear operator.
    eivec: torch.Tensor
        Eigenvectors of the linear operator.

    """
    if mode == "lowest":
        eival = eival[..., :neig]
        eivec = eivec[..., :neig]
    else:
        eival = eival[..., -neig:]
        eivec = eivec[..., -neig:]
    return eival, eivec
