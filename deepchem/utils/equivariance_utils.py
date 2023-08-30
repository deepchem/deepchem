import math
from math import factorial
from typing import Optional, Tuple
from fractions import Fraction
import numpy as np
import torch


def su2_generators(k: int) -> torch.Tensor:
    """Generate the generators of the special unitary group SU(2) in a given representation.

    The function computes the generators of the SU(2) group for a specific representation
    determined by the value of 'k'. These generators are commonly used in the study of
    quantum mechanics, angular momentum, and related areas of physics and mathematics.
    The generators are represented as matrices.

    The SU(2) group is a fundamental concept in quantum mechanics and symmetry theory.
    The generators of the group, denoted as J_x, J_y, and J_z, represent the three
    components of angular momentum operators. These generators play a key role in
    describing the transformation properties of physical systems under rotations.

    The returned tensor contains three matrices corresponding to the x, y, and z generators,
    usually denoted as J_x, J_y, and J_z. These matrices form a basis for the Lie algebra
    of the SU(2) group.

    In linear algebra, specifically within the context of quantum mechanics, lowering and
    raising operators are fundamental concepts that play a crucial role in altering the
    eigenvalues of certain operators while acting on quantum states. These operators are
    often referred to collectively as "ladder operators."

    A lowering operator is an operator that, when applied to a quantum state, reduces the
    eigenvalue associated with a particular observable. In the context of SU(2), the lowering
    operator corresponds to J_-.

    Conversely, a raising operator is an operator that increases the eigenvalue of an
    observable when applied to a quantum state. In the context of SU(2), the raising operator
    corresponds to J_+.

    The z-generator matrix represents the component of angular momentum along the z-axis,
    often denoted as J_z. It commutes with both J_x and J_y and is responsible for quantizing
    the angular momentum.

    Note that the dimensions of the returned tensor will be (3, 2j+1, 2j+1), where each matrix
    has a size of (2j+1) x (2j+1).
    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.

    Returns
    -------
    torch.Tensor
        A stack of three SU(2) generators, corresponding to J_x, J_z, and J_y.

    Notes
    -----
    A generating set of a group is a subset $S$ of the group $G$ such that every element
    of $G$ can be expressed as a combination (under the group operation) of finitely many
    elements of the subset $S$ and their inverses.

    The special unitary group $SU_n(q)$ is the set of $n*n$ unitary matrices with determinant
    +1. $SU(2)$ is homeomorphic with the orthogonal group $O_3^+(2)$. It is also called the
    unitary unimodular group and is a Lie group.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Ladder_operator
    .. [2] https://en.wikipedia.org/wiki/Special_unitary_group#The_group_SU(2)
    .. [3] https://en.wikipedia.org/wiki/Generating_set_of_a_group
    .. [4] https://mathworld.wolfram.com/SpecialUnitaryGroup

    Examples
    --------
    >>> su2_generators(1)
    tensor([[[ 0.0000+0.0000j,  0.7071+0.0000j,  0.0000+0.0000j],
             [-0.7071+0.0000j,  0.0000+0.0000j,  0.7071+0.0000j],
             [ 0.0000+0.0000j, -0.7071+0.0000j,  0.0000+0.0000j]],
    <BLANKLINE>
            [[-0.0000-1.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
             [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+1.0000j]],
    <BLANKLINE>
            [[ 0.0000-0.0000j,  0.0000+0.7071j,  0.0000-0.0000j],
             [ 0.0000+0.7071j,  0.0000-0.0000j,  0.0000+0.7071j],
             [ 0.0000-0.0000j,  0.0000+0.7071j,  0.0000-0.0000j]]])
    """
    # Generate the raising operator matrix
    m = torch.arange(-k, k)
    raising = torch.diag(-torch.sqrt(k * (k + 1) - m * (m + 1)), diagonal=-1)

    # Generate the lowering operator matrix
    m = torch.arange(-k + 1, k + 1)
    lowering = torch.diag(torch.sqrt(k * (k + 1) - m * (m - 1)), diagonal=1)

    # Generate the z-generator matrix
    m = torch.arange(-k, k + 1)
    z_generator = torch.diag(1j * m)

    # Combine the matrices to form the x, z, and y generators
    x_generator = 0.5 * (raising + lowering)  # x (usually)
    y_generator = -0.5j * (raising - lowering)  # -y (usually)

    # Stack the generators along the first dimension to create a tensor
    generators = torch.stack([x_generator, z_generator, y_generator], dim=0)

    return generators


def change_basis_real_to_complex(
        k: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None) -> torch.Tensor:
    r"""Construct a transformation matrix to change the basis from real to complex spherical harmonics.

    This function constructs a transformation matrix Q that converts real spherical
    harmonics into complex spherical harmonics.
    It operates on the basis functions $Y_{\ell m}$ and $Y_{\ell}^{m}$, and accounts
    for the relationship between the real and complex forms of these harmonics
    as defined in the provided mathematical expressions.

    The resulting transformation matrix Q is used to change the basis of vectors or tensors of real spherical harmonics to
    their complex counterparts.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    dtype : torch.dtype, optional
        The data type for the output tensor. If not provided, the
        function will infer it. Default is None.
    device : torch.device, optional
        The device where the output tensor will be placed. If not provided,
        the function will use the default device. Default is None.

    Returns
    -------
    torch.Tensor
        A transformation matrix Q that changes the basis from real to complex spherical harmonics.

    Notes
    -----
    Spherical harmonics Y_l^m are a family of functions that are defined on the surface of a
    unit sphere. They are used to represent various physical and mathematical phenomena that
    exhibit spherical symmetry. The indices l and m represent the degree and order of the
    spherical harmonics, respectively.

    The conversion from real to complex spherical harmonics is achieved by applying specific
    transformation coefficients to the real-valued harmonics. These coefficients are derived
    from the properties of spherical harmonics.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    Examples
    --------
    # The transformation matrix generated is used to change the basis of a vector of
    # real spherical harmonics with representation index 1 to complex spherical harmonics.
    >>> change_basis_real_to_complex(1)
    tensor([[-0.7071+0.0000j,  0.0000+0.0000j,  0.0000-0.7071j],
            [ 0.0000+0.0000j,  0.0000-1.0000j,  0.0000+0.0000j],
            [-0.7071+0.0000j,  0.0000+0.0000j,  0.0000+0.7071j]])
    """
    q = torch.zeros((2 * k + 1, 2 * k + 1), dtype=torch.complex128)

    # Construct the transformation matrix Q for m in range(-k, 0)
    for m in range(-k, 0):
        q[k + m, k + abs(m)] = 1 / 2**0.5
        q[k + m, k - abs(m)] = -1j / 2**0.5

    # Set the diagonal elements for m = 0
    q[k, k] = 1

    # Construct the transformation matrix Q for m in range(1, k + 1)
    for m in range(1, k + 1):
        q[k + m, k + abs(m)] = (-1)**m / 2**0.5
        q[k + m, k - abs(m)] = 1j * (-1)**m / 2**0.5

    # Apply the factor of (-1j)**k to make the Clebsch-Gordan coefficients real
    q = (-1j)**k * q

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


def so3_generators(k: int) -> torch.Tensor:
    """Construct the generators of the SO(3) Lie algebra for a given quantum angular momentum.

    The function generates the generators of the special orthogonal group SO(3), which represents the group
    of rotations in three-dimensional space. Its Lie algebra, which consists of the generators of
    infinitesimal rotations, is often used in physics to describe angular momentum operators.
    The generators of the Lie algebra can be related to the SU(2) group, and this function uses
    a transformation to convert the SU(2) generators to the SO(3) basis.

    The primary significance of the SO(3) group lies in its representation of three-dimensional
    rotations. Each matrix in SO(3) corresponds to a unique rotation, capturing the intricate
    ways in which objects can be oriented in 3D space. This concept finds application in
    numerous fields, ranging from physics to engineering.

    Parameters
    ----------
     k : int
        The representation index, which determines the order of the representation.

    Returns
    -------
    torch.Tensor
        A stack of three SO(3) generators, corresponding to J_x, J_z, and J_y.

    Notes
    -----
    The special orthogonal group $SO_n(q)$ is the subgroup of the elements of general orthogonal
    group $GO_n(q)$ with determinant 1. $SO_3$ (often written $SO(3)$) is the rotation group
    for three-dimensional space.

    These matrices are orthogonal, which means their rows and columns form mutually perpendicular
    unit vectors. This preservation of angles and lengths makes orthogonal matrices fundamental
    in various mathematical and practical applications.

    The "special" part of $SO(3)$ refers to the determinant of these matrices being $+1$. The
    determinant is a scalar value that indicates how much a matrix scales volumes.
    A determinant of $+1$ ensures that the matrix represents a rotation in three-dimensional
    space without involving any reflection or scaling operations that would reverse the orientation of space.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Special_orthogonal_group
    .. [2] https://en.wikipedia.org/wiki/3D_rotation_group#Connection_between_SO(3)_and_SU(2)
    .. [3] https://www.pas.rochester.edu/assets/pdf/undergraduate/su-2s_double_covering_of_so-3.pdf

    Examples
    --------
    >>> so3_generators(1)
    tensor([[[ 0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000, -1.0000],
             [ 0.0000,  1.0000,  0.0000]],
    <BLANKLINE>
            [[ 0.0000,  0.0000,  1.0000],
             [ 0.0000,  0.0000,  0.0000],
             [-1.0000,  0.0000,  0.0000]],
    <BLANKLINE>
            [[ 0.0000, -1.0000,  0.0000],
             [ 1.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000]]])
    """
    # Get the SU(2) generators for the given quantum angular momentum (spin) value.
    X = su2_generators(k)

    # Get the transformation matrix to change the basis from real to complex spherical harmonics.
    Q = change_basis_real_to_complex(k)

    # Convert the SU(2) generators to the SO(3) basis using the transformation matrix Q.
    # X represents the SU(2) generators, and Q is the transformation matrix from real to complex spherical harmonics.
    # The resulting X matrix will be the SO(3) generators in the complex basis.
    X = torch.conj(Q.T) @ X @ Q

    # Return the real part of the SO(3) generators to ensure they are purely real.
    return torch.real(X)


def wigner_D(k: int, alpha: torch.Tensor, beta: torch.Tensor,
             gamma: torch.Tensor) -> torch.Tensor:
    """Wigner D matrix representation of the SO(3) rotation group.

    The function computes the Wigner D matrix representation of the SO(3) rotation group
    for a given representation index 'k' and rotation angles 'alpha', 'beta', and 'gamma'.
    The resulting matrix satisfies properties of the SO(3) group representation.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    alpha : torch.Tensor
        Rotation angles (in radians) around the Y axis, applied third.
    beta : torch.Tensor
        Rotation angles (in radians) around the X axis, applied second.
    gamma : torch.Tensor)
        Rotation angles (in radians) around the Y axis, applied first.

    Returns
    -------
    torch.Tensor
        The Wigner D matrix of shape (2j+1, 2j+1).

    Notes
    -----
    The Wigner D-matrix is a unitary matrix in an irreducible representation
    of the groups SU(2) and SO(3).

    The Wigner D-matrix is used in quantum mechanics to describe the action
    of rotations on states of particles with angular momentum. It is a key
    concept in the representation theory of the rotation group SO(3), and
    it plays a crucial role in various physical contexts.

    Examples
    --------
    >>> k = 1
    >>> alpha = torch.tensor([0.1, 0.2])
    >>> beta = torch.tensor([0.3, 0.4])
    >>> gamma = torch.tensor([0.5, 0.6])
    >>> wigner_D_matrix = wigner_D(k, alpha, beta, gamma)
    >>> wigner_D_matrix
    tensor([[[ 0.8275,  0.0295,  0.5607],
             [ 0.1417,  0.9553, -0.2593],
             [-0.5433,  0.2940,  0.7863]],
    <BLANKLINE>
            [[ 0.7056,  0.0774,  0.7044],
             [ 0.2199,  0.9211, -0.3214],
             [-0.6737,  0.3816,  0.6329]]])
    """
    # Ensure that alpha, beta, and gamma have the same shape for broadcasting.
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)

    # Ensure the angles are within the range [0, 2*pi) using modulo.
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    # Get the SO(3) generators for the given quantum angular momentum (spin) value 'k'.
    X = so3_generators(k)

    # Calculate the Wigner D matrix using the matrix exponential of the generators
    # and the rotation angles alpha, beta, and gamma in the appropriate order.
    D_matrix = torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(
        beta * X[0]) @ torch.matrix_exp(gamma * X[1])
    return D_matrix


def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the commutator of two matrices.

    Parameters
    ----------
    A : torch.Tensor
        The first matrix.
    B : torch.Tensor
        The second matrix.

    Returns
    -------
    torch.Tensor
        The commutator of the two matrices.

    Examples
    --------
    >>> A = torch.tensor([[1, 2], [3, 4]])
    >>> B = torch.tensor([[5, 6], [7, 8]])
    >>> commutator(A, B)
    tensor([[ -4, -12],
            [ 12,   4]])
    """
    return torch.matmul(A, B) - torch.matmul(B, A)


def calculate_clebsch_gordan_coefficient(j1_m1: Tuple[float, float],
                                         j2_m2: Tuple[float, float],
                                         j3_m3: Tuple[float, float]) -> float:
    """Calculate the Clebsch-Gordan coefficient for coupling two angular
    momenta (j1,m1) and (j2,m2) to give (j3,m3).

    The Clebsch-Gordan coefficients are numbers used in quantum mechanics
    for combining angular momenta. They represent the expansion of total
    angular momentum states in a basis of uncoupled tensor products.

    Mathematically, they're crucial in representation theory, especially
    for compact Lie groups. They help break down a tensor product of
    irreducible representations into its constituent parts.

    Parameters:
    -----------
    j1_m1 : Tuple[float, float]
        Tuple representing angular momentum and its z-component for particle 1.
    j2_m2 : Tuple[float, float]
        Tuple representing angular momentum and its z-component for particle 2.
    j3_m3 : Tuple[float, float]
        Tuple representing angular momentum and its z-component for the coupled system.

    Returns:
    --------
    cg_coefficient : float
        Clebsch-Gordan coefficient for the given coupling.

    References:
    -----------
    .. [1] https://en.wikipedia.org/wiki/Clebsch_Gordan_coefficients
    .. [2] https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients#Relation_to_Wigner_3-j_symbols
    .. [3] https://qutip.org/docs/3.1.0/modules/qutip/utilities.html#clebsch_gordan

    Examples:
    ---------
    >>> j1_m1 = (1, 1)
    >>> j2_m2 = (1, -1)
    >>> j3_m3 = (1, 0)
    >>> calculate_clebsch_gordan_coefficient(j1_m1, j2_m2, j3_m3)
    0.7071067811865476
    """
    # Unpack input tuples to extract angular momentum and z-component values
    j1, m1 = j1_m1
    j2, m2 = j2_m2
    j3, m3 = j3_m3

    # Check if z-components can be coupled to result in desired z-component
    if m3 != m1 + m2:
        return 0.0

    # Calculate the allowed range of summation indices
    vmin = max(-j1 + j2 + m3, -j1 + m1, 0)
    vmax = min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3)

    # Define a function to calculate the factorial of a rounded number
    def factorial_round(n: float) -> float:
        return factorial(round(n))

    # Calculate the overall Clebsch-Gordan coefficient (C)
    C = ((2.0 * j3 + 1.0) * Fraction(
        factorial_round(j3 + j1 - j2) * factorial_round(j3 - j1 + j2) *
        factorial_round(j1 + j2 - j3) * factorial_round(j3 + m3) *
        factorial_round(j3 - m3),
        factorial_round(j1 + j2 + j3 + 1) * factorial_round(j1 - m1) *
        factorial_round(j1 + m1) * factorial_round(j2 - m2) *
        factorial_round(j2 + m2),
    ))**0.5

    # Calculate the summation term (S) in the Clebsch-Gordan formula
    S = sum((-1)**int(v + j2 + m2) * Fraction(
        factorial_round(j2 + j3 + m1 - v) * factorial_round(j1 - m1 + v),
        factorial_round(v) * factorial_round(j3 - j1 + j2 - v) *
        factorial_round(j3 + m3 - v) * factorial_round(v + j1 - j2 - m3))
            for v in range(int(vmin),
                           int(vmax) + 1))

    # Multiply the calculated coefficient (C) by the summation term (S)
    C *= S

    # Return the Clebsch-Gordan coefficient as a float
    return float(C)


def calculate_su2_clebsch_gordan(j1: float, j2: float,
                                 j3: float) -> torch.tensor:
    """Calculate the Clebsch-Gordan coefficients for SU(2) coupling.

    Calculates the Clebsch-Gordan coefficients for coupling two angular
    momenta (j1) and (j2) to give (j3). This function computes the Clebsch-Gordan
    coefficients for a range of possible magnetic quantum numbers (m1, m2, m3),
    based on the triangle inequalities for angular momenta.

    Parameters:
    -----------
    j1 : float
        Angular momentum for particle 1.
    j2 : float
        Angular momentum for particle 2.
    j3 : float
        Angular momentum for the coupled system.

    Returns:
    --------
    cg_matrix : torch.tensor
        Matrix of Clebsch-Gordan coefficients for the given coupling.
        The dimensions of the matrix are (2*j1+1) x (2*j2+1) x (2*j3+1).

    See Also:
    ---------
    calculate_clebsch_gordan_coefficient

    Notes:
    ------
    The Clebsch-Gordan coefficients describe the coupling of two angular momenta (j1 and j2)
    to form a resultant angular momentum (j3). This function calculates the Clebsch-Gordan
    coefficients for valid combinations of quantum numbers (m1, m2, m3), which satisfy
    the triangle inequalities for angular momenta.

    The matrix dimensions are based on the quantum numbers j1, j2, and j3, and the
    Clebsch-Gordan coefficients are computed for valid combinations within this matrix.

    References:
    -----------
    .. [1] https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients

    Examples:
    ---------
    >>> cg_matrix = calculate_su2_clebsch_gordan(1.0, 1.0, 1.0)
    >>> print(cg_matrix)
    tensor([[[ 0.0000,  0.0000,  0.0000],
             [-0.7071,  0.0000,  0.0000],
             [ 0.0000, -0.7071,  0.0000]],
    <BLANKLINE>
            [[ 0.7071,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000, -0.7071]],
    <BLANKLINE>
            [[ 0.0000,  0.7071,  0.0000],
             [ 0.0000,  0.0000,  0.7071],
             [ 0.0000,  0.0000,  0.0000]]], dtype=torch.float64)
    """
    # Calculate the matrix dimensions based on quantum numbers.
    matrix_shape = (int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1))
    cg_matrix = np.zeros(matrix_shape, dtype=np.float64)

    for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
        for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
            m3 = m1 + m2
            if abs(m3) <= j3:
                coefficient = calculate_clebsch_gordan_coefficient(
                    (j1, m1), (j2, m2), (j3, m3))
                cg_matrix[int(j1 + m1),
                          int(j2 + m2),
                          int(j3 + m3)] = coefficient

    cg_matrix = torch.from_numpy(cg_matrix)
    return cg_matrix


def calculate_so3_clebsch_gordan(j1: int, j2: int, j3: int) -> torch.Tensor:
    """
    Calculate the Clebsch-Gordan coefficients for coupling three angular momenta (j1, j2, j3) in SO(3).

    Parameters
    ----------
    j1 : int
        Angular momentum value for the first particle.
    j2 : int
        Angular momentum value for the second particle.
    j3 : int
        Angular momentum value for the coupled system.

    Returns
    -------
    C : torch.Tensor
        Clebsch-Gordan coefficients for the given angular momenta.

    Notes
    -----
    This function calculates Clebsch-Gordan coefficients for coupling three angular momenta (j1, j2, j3)
    using the provided basis transformation matrices and the pre-calculated SU(2) Clebsch-Gordan coefficients.

    The resulting coefficients are normalized and returned as a torch tensor.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients#Relation_to_Wigner_3-j_symbols

    Examples
    --------
    >>> j1 = 1
    >>> j2 = 1
    >>> j3 = 1
    >>> coefficients = calculate_so3_clebsch_gordan(j1, j2, j3)
    >>> print(coefficients)
    tensor([[[ 0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.4082],
             [ 0.0000, -0.4082,  0.0000]],
    <BLANKLINE>
            [[ 0.0000,  0.0000, -0.4082],
             [ 0.0000,  0.0000,  0.0000],
             [ 0.4082,  0.0000,  0.0000]],
    <BLANKLINE>
            [[ 0.0000,  0.4082,  0.0000],
             [-0.4082,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000]]], dtype=torch.float64)
    """
    # Transform real basis to complex basis
    Q1 = change_basis_real_to_complex(j1, dtype=torch.complex128)
    Q2 = change_basis_real_to_complex(j2, dtype=torch.complex128)
    Q3 = change_basis_real_to_complex(j3, dtype=torch.complex128)

    # Calculate SU(2) Clebsch-Gordan coefficients and convert to complex dtype
    C_su2 = calculate_su2_clebsch_gordan(j1, j2, j3).to(dtype=torch.complex128)

    # Calculate final Clebsch-Gordan coefficients using Einstein summation (torch.einsum)
    # The einsum expression "ij,kl,mn,ikn->jlm" performs the following operations:
    # - Contracts index 'i' of Q1 with index 'k' of Q3, and index 'k' of Q3 with index 'i' of C_su2
    # - Contracts index 'j' of Q1 with index 'm' of Q3, and index 'm' of Q3 with index 'j' of C_su2
    # - Contracts index 'k' of Q2 with index 'n' of Q3, and index 'i' of C_su2 with index 'k' of Q2
    # The result is a tensor with dimensions (j, l, m) containing the Clebsch-Gordan coefficients.
    C = torch.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, torch.conj(Q3.T),
                     C_su2.to(torch.complex128)).real

    # Normalize the coefficients
    C /= torch.norm(C)

    return C
