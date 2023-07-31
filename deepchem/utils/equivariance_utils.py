import math
import torch
from typing import Optional


def su2_generators(j: int) -> torch.Tensor:
    """Get the generators of the SU(2) Lie algebra for a given quantum angular momentum.

    Parameters
    ----------
        j (int): The quantum angular momentum (spin) value.

    Returns
    -------
        torch.Tensor: A stack of three SU(2) generators, corresponding to J_x, J_z, and J_y.
    """
    # Generate a range of quantum angular momentum projections along the z-axis from -j to j-1.
    m = torch.arange(-j, j)

    # Construct the raising operator (J_+) of the SU(2) algebra.
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    # Generate a range of quantum angular momentum projections along the z-axis from -j+1 to j.
    m = torch.arange(-j + 1, j + 1)

    # Construct the lowering operator (J_-) of the SU(2) algebra.
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    # Generate a range of quantum angular momentum projections along the z-axis from -j to j.
    m = torch.arange(-j, j + 1)

    # Stack the three generators (J_x, J_z, and J_y) of the SU(2) algebra.
    generators = torch.stack(
        [
            0.5 * (raising + lowering),  # J_x
            torch.diag(1j * m),  # J_z
            -0.5j * (raising - lowering),  # -J_y
        ],
        dim=0,
    )
    return generators


def change_basis_real_to_complex(
        j: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None) -> torch.Tensor:
    """Construct a transformation matrix to change the basis from real to complex spherical harmonics.

    The function generates a matrix representing the transformation between the basis
    of real and complex spherical harmonics. It is used to convert the representation of
    the angular momentum states from the real basis (used in SO(3) group) to the complex
    basis (used in SU(2) group).

    Parameters
    ----------
        j (int): The quantum angular momentum (spin) value.
        dtype (torch.dtype, optional): The data type for the output tensor. If not provided, the
            function will use torch.complex128. Default is None.
        device (torch.device, optional): The device where the output tensor will be placed.
            If not provided, the function will use the default device. Default is None.

    Returns
    -------
        torch.Tensor: A transformation matrix Q that changes the basis from real to complex spherical harmonics.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    q = torch.zeros((2 * j + 1, 2 * j + 1), dtype=torch.complex128)

    # Construct the transformation matrix Q for m in range(-j, 0)
    for m in range(-j, 0):
        q[j + m, j + abs(m)] = 1 / 2**0.5
        q[j + m, j - abs(m)] = -1j / 2**0.5

    # Set the diagonal elements for m = 0
    q[j, j] = 1

    # Construct the transformation matrix Q for m in range(1, j + 1)
    for m in range(1, j + 1):
        q[j + m, j + abs(m)] = (-1)**m / 2**0.5
        q[j + m, j - abs(m)] = 1j * (-1)**m / 2**0.5

    # Apply the factor of (-1j)**j to make the Clebsch-Gordan coefficients real
    q = (-1j)**j * q

    # Handle dtype and device options
    if dtype is None:
        default_type = torch.empty(0).dtype
        if default_type == torch.float32:
            dtype = torch.complex64
        elif default_type == torch.float64:
            dtype = torch.complex128
    if device is None:
        device = torch.empty(0).device

    # Ensure the tensor is contiguous and on the specified device
    return q.to(dtype=dtype,
                device=device,
                copy=True,
                memory_format=torch.contiguous_format)


def so3_generators(j: int) -> torch.Tensor:
    """Construct the generators of the SO(3) Lie algebra for a given quantum angular momentum.

    The function generates the generators of the SO(3) Lie algebra by converting the SU(2) generators
    to the SO(3) basis using the transformation matrix from real to complex spherical harmonics.

    Parameters
    ----------
        j (int): The quantum angular momentum (spin) value.

    Returns
    -------
        torch.Tensor: A stack of three SO(3) generators, corresponding to J_x, J_z, and J_y.

    References
    ----------
    .. [1] https://www.pas.rochester.edu/assets/pdf/undergraduate/su-2s_double_covering_of_so-3.pdf

    """
    # Get the SU(2) generators for the given quantum angular momentum (spin) value.
    X = su2_generators(j)

    # Get the transformation matrix to change the basis from real to complex spherical harmonics.
    Q = change_basis_real_to_complex(j)

    # Convert the SU(2) generators to the SO(3) basis using the transformation matrix Q.
    # X represents the SU(2) generators, and Q is the transformation matrix from real to complex spherical harmonics.
    # The resulting X matrix will be the SO(3) generators in the complex basis.
    X = torch.conj(Q.T) @ X @ Q

    # Return the real part of the SO(3) generators to ensure they are purely real.
    return torch.real(X)


def wigner_D(j: int, alpha: torch.Tensor, beta: torch.Tensor,
             gamma: torch.Tensor) -> torch.Tensor:
    """Wigner D matrix representation of the SO(3) rotation group.

    The function computes the Wigner D matrix representation of the SO(3) rotation group
    for a given quantum angular momentum 'j' and rotation angles 'alpha', 'beta', and 'gamma'.
    The resulting matrix satisfies properties of the SO(3) group representation.

    Parameters
    ----------
        j (int): The quantum angular momentum (spin) value.
        alpha (torch.Tensor): Rotation angles (in radians) around the Y axis, applied third.
        beta (torch.Tensor): Rotation angles (in radians) around the X axis, applied second.
        gamma (torch.Tensor): Rotation angles (in radians) around the Y axis, applied first.

    Returns
    -------
        torch.Tensor: The Wigner D matrix of shape (2l+1, 2l+1).
    """
    # Ensure that alpha, beta, and gamma have the same shape for broadcasting.
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)

    # Ensure the angles are within the range [0, 2*pi) using modulo.
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    # Get the SO(3) generators for the given quantum angular momentum (spin) value 'j'.
    X = so3_generators(j)

    # Calculate the Wigner D matrix using the matrix exponential of the generators
    # and the rotation angles alpha, beta, and gamma in the appropriate order.
    D_matrix = torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(
        beta * X[0]) @ torch.matrix_exp(gamma * X[1])
    return D_matrix
