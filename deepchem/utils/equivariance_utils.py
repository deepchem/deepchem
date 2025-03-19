import math
from typing import Optional, List, Dict, Tuple
import torch
import numpy as np


def get_basis(G,
              max_degree: int,
              compute_gradients: bool = False) -> Dict[str, torch.Tensor]:
    """
    Precompute the SE(3)-equivariant weight basis, \( W_J^{lk}(x) \).

    This function computes the equivariant weight basis used in SE(3)-equivariant
    convolutions, enabling the model to learn **rotation-equivariant filters.

    It is **called internally** by `get_basis_and_r()`, which computes **both**:
    - **Equivariant weight basis** \( W_J^{lk}(x) \).
    - **Inter-nodal distances** \( r_{ij} \) (used in attention & convolution).

    ---
    ### **Mathematical Background**
    This function follows Equation (8) from [SE(3)-Transformer paper](https://arxiv.org/pdf/2006.10503.pdf):

    \[
    W_J^{lk}(x) = Q_J \cdot Y_J(x)
    \]

    - \( Y_J(x) \): **Spherical Harmonic Basis**
    - \( Q_J \): **Basis transformation matrix**
    - \( J \): **Angular momentum index**
    - \( d_{in}, d_{out} \): **Feature degrees**

    This function precomputes basis functions for SE(3)-equivariant convolutions by calcutating spherical harmonic projections.

    Parameters
    ----------
    G:  dgl.DGLGraph
        DGL graph where `G.edata['d']` stores edge displacement vectors.
    max_degree: int
        Maximum degree (`l_max`) of equivariant tensors.
    compute_gradients: `bool`, optional, default=`False`
        If `True`: Enables gradient tracking for backpropagation.
        If `False`: Uses `torch.no_grad()` for efficiency.

    Returns
    -------
    basis: Dict[str, torch.Tensor]
        Dictionary mapping `'<d_in>,<d_out>'`: precomputed basis tensor.
        Shape: `(batch_size, 1, 2*d_out+1, 1, 2*d_in+1, num_bases)`

    Example
    -------

    >>> import torch
    >>> import dgl
    >>> from deepchem.utils.equivariance_utils import get_basis
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True, embeded=True)
    >>> features = featurizer.featurize([mol])[0]
    >>> G = dgl.graph((features.edge_index[0], features.edge_index[1]))
    >>> G.ndata['f'] = torch.tensor(features.node_features, dtype=torch.float32).unsqueeze(-1)
    >>> G.ndata['x'] = torch.tensor(features.positions, dtype=torch.float32)
    >>> G.edata['d'] = torch.tensor(features.edge_features, dtype=torch.float32)
    >>> G.edata['w'] = torch.tensor(features.edge_weights, dtype=torch.float32)
    >>> basis = get_basis(G, max_degree=2)
    >>> print(basis.keys())
    dict_keys(['0,0', '0,1', '0,2', '1,0', '1,1', '1,2', '2,0', '2,1', '2,2'])
    """
    # Disable gradients unless explicitly enabled
    context = torch.enable_grad() if compute_gradients else torch.no_grad()

    with context:
        cloned_d = torch.clone(G.edata['d'])
        if G.edata['d'].requires_grad:
            cloned_d.requires_grad_()

        # Compute relative positional encodings in spherical coordinates
        r_ij = get_spherical_from_cartesian(cloned_d)

        # Compute Spherical Harmonics (Y_J) for all degrees up to 2*max_degree
        Y = precompute_sh(r_ij, 2 * max_degree)
        device = Y[0].device

        basis = {}
        for d_in in range(max_degree + 1):
            for d_out in range(max_degree + 1):
                K_Js = []
                for J in range(abs(d_in - d_out), d_in + d_out + 1):
                    # Compute basis transformation matrix Q_J
                    Q_J = basis_transformation_Q_J(J, d_in,
                                                   d_out).float().T.to(device)

                    # Apply projection using spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape for broadcasting
                size = (-1, 1, 2 * d_out + 1, 1, 2 * d_in + 1,
                        2 * min(d_in, d_out) + 1)
                basis[f"{d_in},{d_out}"] = torch.stack(K_Js, -1).view(*size)

        return basis


def get_r(G) -> torch.Tensor:
    """
    Compute inter-nodal distances for a given DGL graph.

    This function computes the Euclidean distance between connected nodes
    based on edge feature `d`, which represents relative displacements.

    Parameters
    ----------
    G : dgl.DGLGraph
        The input graph where `G.edata['d']` contains edge displacement vectors.

    Returns
    -------
    torch.Tensor
        A tensor containing the computed inter-nodal distances for each edge,
        with shape `(num_edges, 1)`.

    Example
    -------
    >>> import torch
    >>> import dgl
    >>> from deepchem.utils.equivariance_utils import get_r
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True, embeded=True)
    >>> features = featurizer.featurize([mol])[0]
    >>> G = dgl.graph((features.edge_index[0], features.edge_index[1]))
    >>> G.ndata['f'] = torch.tensor(features.node_features, dtype=torch.float32).unsqueeze(-1)
    >>> G.ndata['x'] = torch.tensor(features.positions, dtype=torch.float32)
    >>> G.edata['d'] = torch.tensor(features.edge_features, dtype=torch.float32)
    >>> G.edata['w'] = torch.tensor(features.edge_weights, dtype=torch.float32)
    >>> # Compute internodal distances
    >>> r = get_r(G)
    >>> print(r.shape)  # (num_edges, 1)
    torch.Size([6, 1])
    >>> print(r)
    tensor([[1.5317],
            [2.3267],
            [1.5317],
            [1.4096],
            [2.3267],
            [1.4096]])
    """
    cloned_d = torch.clone(G.edata['d'])
    if G.edata['d'].requires_grad:
        cloned_d.requires_grad_()
    return torch.sqrt(torch.sum(cloned_d**2, -1, keepdim=True))


def get_equivariant_basis_and_r(
    G,
    max_degree: int,
    compute_gradients: bool = False
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Compute equivariant weight basis and inter-nodal distances for SE(3)-Transformers.

    This function computes:
    1. **Equivariant weight basis**: Required for SE(3)-equivariant convolutions.
    2. **Inter-nodal distances**: Used in attention mechanisms and basis function computation.

    This should be **called once per forward pass** to compute shared equivariant representations.

    Parameters
    ----------
    G : dgl.DGLGraph
        A DGL graph containing edge features.
    max_degree : int
        The maximum degree of the SE(3) tensor representation.
    compute_gradients : bool, optional (default=False)
        If True, enables gradient computation for the basis.

    Returns
    -------
    Tuple[Dict[str, torch.Tensor], torch.Tensor]
        - **basis (dict[str, torch.Tensor])**: Dictionary of equivariant bases,
          indexed by `'<d_in><d_out>'`, where `d_in` and `d_out` are feature degrees.
        - **r (torch.Tensor)**: Inter-nodal distance tensor with shape `(num_edges, 1)`.

    Example
    -------
    >>> import torch
    >>> import dgl
    >>> from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=True, embeded=True)
    >>> features = featurizer.featurize([mol])[0]
    >>> G = dgl.graph((features.edge_index[0], features.edge_index[1]))
    >>> G.ndata['f'] = torch.tensor(features.node_features, dtype=torch.float32).unsqueeze(-1)
    >>> G.ndata['x'] = torch.tensor(features.positions, dtype=torch.float32)
    >>> G.edata['d'] = torch.tensor(features.edge_features, dtype=torch.float32)
    >>> G.edata['w'] = torch.tensor(features.edge_weights, dtype=torch.float32)
    >>> # Compute basis and distances
    >>> basis, r = get_equivariant_basis_and_r(G, max_degree=2)
    >>> print(r.shape)  # Expected: (num_edges, 1)
    torch.Size([6, 1])
    >>> print(basis.keys())  # Expected: dict
    dict_keys(['0,0', '0,1', '0,2', '1,0', '1,1', '1,2', '2,0', '2,1', '2,2'])
    """
    basis = get_basis(G, max_degree, compute_gradients)
    r = get_r(G)
    return basis, r


def semifactorial(x: int) -> float:
    """
    Compute the semifactorial function x!!, defined as:
    x!! = x * (x-2) * (x-4) * ...
    Parameters
    ----------
    x : int
        A positive integer.
    Returns
    -------
    float
        The value of x!!, computed iteratively.
    Examples
    --------
    >>> semifactorial(5)
    15.0
    >>> semifactorial(6)
    48.0
    """
    # edge cases
    if x == 0 or x == 1:
        return 1.0

    y = 1.0
    for n in range(x, 1, -2):
        y *= n
    return y


def pochhammer(x: int, k: int) -> float:
    """
    Compute the Pochhammer symbol (x)_k , defined as:
    (x)_k = x * (x+1) * (x+2) * ... * (x+k-1).
    Parameters
    ----------
    x : int
        The starting integer of the sequence.
    k : int
        The number of terms in the product.
    Returns
    -------
    float
        The Pochhammer symbol value.
    Examples
    --------
    >>> pochhammer(3, 4)
    360.0
    >>> pochhammer(5, 2)
    30.0
    """
    # handle edge case (k=0)
    if k == 0:
        return 1.0

    xf = float(x)
    for n in range(x + 1, x + k):
        xf *= n
    return xf


def lpmv(d: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the associated Legendre function P_l^m(x), including
    the Condon-Shortley phase. The implementation includes the base case
    for P_m^m(x), P_{m+1}^m(x), and uses a recurrence relation
    for higher degrees.
    Parameters
    ----------
    l : int
        Degree of the Legendre polynomial.
    m : int
        Order of the Legendre polynomial.
    x : torch.Tensor
        A tensor of shape (N,) representing the input values for x,
        where ( -1 <= x <= 1).
    Returns
    -------
    torch.Tensor
        A tensor of shape (N,) containing the values of P_l^m(x).
    Examples
    --------
    >>> lpmv(2, 1, torch.tensor([0.5]))
    tensor([-1.2990])
    """
    m_abs = abs(m)
    if m_abs > d:
        return torch.zeros_like(x)

    # base case: P_m^m(x)
    p_mm = ((-1)**m_abs * semifactorial(2 * m_abs - 1)) * torch.pow(
        1 - x * x, m_abs / 2)

    if d == m_abs:
        return p_mm

    # Compute P_{m+1}^m(x)
    p_m1m = x * (2 * m_abs + 1) * p_mm

    if d == m_abs + 1:
        return p_m1m

    # Recurrence relation for higher degrees (l)
    p_lm = p_m1m.clone()
    for i in range(m_abs + 2, d + 1):
        p_lm = ((2 * i - 1) * x * p_m1m - (i + m_abs - 1) * p_mm) / (i - m_abs)
        p_mm, p_m1m = p_m1m, p_lm

    # Negative values of m
    if m < 0:
        p_lm *= ((-1)**m) * pochhammer(d - m + 1, -2 * m)

    return p_lm


class SphericalHarmonics:
    """
    A class for computing real tesseral spherical harmonics, including
    the Condon-Shortley phase.
    This implementation is inspired by the SE(3)-Transformer library by Fabian Fuchs,
    which provides efficient computation of spherical harmonics and related transformations.
    For more details, see the SE(3)-Transformer repository:
    https://github.com/FabianFuchsML/se3-transformer-public

    Methods
    -------
    get_element(l, m, theta, phi)
        Compute the tesseral spherical harmonic Y_l^m(theta, phi).
    get(l, theta, phi)
        Compute all spherical harmonics of degree l for given angles.
    """

    def __init__(self):
        self.leg = {}

    def clear(self) -> None:
        """Clear cached Legendre polynomial values to save RAM memory."""
        self.leg = {}

    def get_element(self, d: int, m: int, theta: torch.Tensor,
                    phi: torch.Tensor) -> torch.Tensor:
        """
        Compute a single tesseral spherical harmonic Y_l^m(theta, phi).
        Parameters
        ----------
        l : int
            Degree of the spherical harmonic.
        m : int
            Order of the spherical harmonic.
        theta : torch.Tensor
            Tensor of polar angles (collatitude) in radians.
        phi : torch.Tensor
            Tensor of azimuthal angles (longitude) in radians.
        Returns
        -------
        torch.Tensor
            Tensor of the spherical harmonic values, same shape as `theta`.
        Examples
        --------
        >>> theta = torch.tensor([0.0, math.pi / 2])
        >>> phi = torch.tensor([0.0, math.pi])
        >>> SphericalHarmonics().get_element(1, 0, theta, phi)
        tensor([ 4.8860e-01, -2.1357e-08])
        """

        N = math.sqrt((2 * d + 1) / (4 * math.pi))
        leg = lpmv(d, abs(m), torch.cos(theta))
        if m == 0:
            return N * leg
        elif m > 0:
            Y = torch.cos(m * phi) * leg
        else:
            Y = torch.sin(abs(m) * phi) * leg
        N *= math.sqrt(2.0 / pochhammer(d - abs(m) + 1, 2 * abs(m)))
        Y *= N
        return Y

    def get(self,
            d: int,
            theta: torch.Tensor,
            phi: torch.Tensor,
            refresh=True) -> torch.Tensor:
        """
        Compute all spherical harmonics of degree l.
        Parameters
        ----------
        l : int
            Degree of the spherical harmonics.
        theta : torch.Tensor
            Tensor of polar angles (collatitude) in radians.
        phi : torch.Tensor
            Tensor of azimuthal angles (longitude) in radians.
        Returns
        -------
        torch.Tensor
            A tensor of shape [*theta.shape, 2 * l + 1].
        Examples
        --------
        >>> theta = torch.tensor([0.0, math.pi / 2])
        >>> phi = torch.tensor([0.0, math.pi])
        >>> SphericalHarmonics().get(1, theta, phi)
        tensor([[-0.0000e+00,  4.8860e-01, -0.0000e+00],
                [ 4.2715e-08, -2.1357e-08,  4.8860e-01]])
        """
        if refresh:
            self.clear()
        results = []
        for m in range(-d, d + 1):
            results.append(self.get_element(d, m, theta, phi))
        return torch.stack(results, dim=-1)


def irr_repr(order: int,
             alpha: float,
             beta: float,
             gamma: float,
             dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Compute the irreducible representation of the special orthogonal group SO(3).
    This function computes the Wigner D-matrix for the given order and angles (alpha, beta, gamma).
    It is compatible with composition and spherical harmonics computations.
    The output is a 2D tensor of shape `(2*order + 1, 2*order + 1)` where each element represents
    the transformation coefficients of the Wigner D-matrix. The values correspond to the rotational
    components for the given order and Euler angles.
    Parameters
    ----------
    order : int
        The order of the representation.
    alpha : float
        The first Euler angle (rotation about the Y axis).
    beta : float
        The second Euler angle (rotation about the X axis).
    gamma : float
        The third Euler angle (rotation about the Y axis).
    dtype : torch.dtype, optional
        The desired data type of the resulting tensor. If None, the default dtype is used.
    Returns
    -------
    torch.Tensor
        The irreducible representation matrix of SO(3) for the specified order and angles.
    Examples
    --------
    >>> irr_repr(1, 0.1, 0.2, 0.3)
    tensor([[ 0.9216,  0.0587,  0.3836],
            [ 0.0198,  0.9801, -0.1977],
            [-0.3875,  0.1898,  0.9021]])
    """
    result = wigner_D(order, torch.tensor(alpha), torch.tensor(beta),
                      torch.tensor(gamma))[0]
    return result.clone().detach() if dtype is None else result.clone().detach(
    ).to(dtype)


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


def get_matrix_kernel(A: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute an orthonormal basis of the kernel (x_1, x_2, ...).

    This function calculates the null space (kernel) of the given matrix A, such that:
        A x_i = 0
        scalar_product(x_i, x_j) = delta_ij

    Parameters
    ----------
    A : torch.Tensor
        The input matrix.
    eps : float, optional
        Tolerance for singular values considered as zero (default is 1e-10).

    Returns
    -------
    torch.Tensor
        A matrix where each row is a basis vector of the kernel of A.

    Examples
    --------
    >>> from deepchem.utils.equivariance_utils import get_matrix_kernel
    >>> A = torch.tensor([[1.0, 2.0, 3.0],
    ...                   [2.0, 4.0, 6.0],
    ...                   [3.0, 6.0, 9.0]])
    >>> get_matrix_kernel(A)
    tensor([[ 0.0000, -0.8321,  0.5547],
            [ 0.9636, -0.1482, -0.2224]])
    """
    _, s, v = torch.svd(A)

    kernel = v.t()[s < eps]
    return kernel


def get_matrices_kernel(As: List[torch.Tensor],
                        eps: float = 1e-10) -> torch.Tensor:
    """
    Compute the common kernel of all the input matrices.

    This function computes the shared null space of a collection of matrices.

    Parameters
    ----------
    As : List[torch.Tensor]
        List of input matrices.
    eps : float, optional
        Tolerance for singular values considered as zero (default is 1e-10).

    Returns
    -------
    torch.Tensor
        A matrix where each row is a basis vector of the common kernel.
    """
    return get_matrix_kernel(torch.cat(As, dim=0), eps)


def basis_transformation_Q_J(J: int,
                             order_in: int,
                             order_out: int,
                             eps: float = 1e-10,
                             num_samples: int = 5,
                             random_angle_higher: float = 6.2,
                             random_angle_lower: float = 0.2) -> torch.Tensor:
    """
    Compute one part of the Q^-1 matrix for the article.

    This function computes the spherical harmonics projection matrix for the
    Sylvester equation in the subspace J needed in for the weight basis in SE(3)-Transformer model.

    References:
    -----------
    - SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
    Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling
    NeurIPS 2020, https://arxiv.org/abs/2006.10503

    Parameters
    ----------
    J : int
        Order of the spherical harmonics.
    order_in : int
        Order of the input representation.
    order_out : int
        Order of the output representation.
    version : int, optional
        Version of the computation (default is 3).
    eps : float, optional
        Tolerance for singular values considered as zero (default is 1e-10).
    num_samples : int, optional
        Number of samples to generate for random angles (default is 5).
    random_angle_higher : float, optional
        Upper limit for generating random angles (default is 6.2).
    random_angle_lower : float, optional
        Lower limit for generating random angles (default is 0.2).

    Returns
    -------
    torch.Tensor
        A tensor of shape [(m_out * m_in), m], where m = 2 * J + 1.

    Examples
    --------
    >>> from deepchem.utils.equivariance_utils import basis_transformation_Q_J
    >>> basis_transformation_Q_J(1, 1, 1).shape
    torch.Size([9, 3])
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    def _R_tensor(a: float, b: float, c: float) -> torch.Tensor:
        """
        Compute the Kronecker product of irreducible representations.

        This function calculates the Kronecker product of two irreducible
        representations (input and output orders) for a given set of rotation angles.

        Parameters
        ----------
        a : float
            Rotation angle around the x-axis.
        b : float
            Rotation angle around the y-axis.
        c : float
            Rotation angle around the z-axis.

        Returns
        -------
        torch.Tensor
            The Kronecker product of the irreducible representations of the
            given rotation angles.

        Examples
        --------
        >>> _R_tensor(1.0, 2.0, 3.0).shape
        torch.Size([...])
        return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))
        """
        return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))

    def _sylvester_submatrix(J: int, a: float, b: float,
                             c: float) -> torch.Tensor:
        """
        Generate the Kronecker product matrix for solving the Sylvester equation in subspace J.

        This function constructs a Kronecker product matrix that helps solve
        the Sylvester equation for the given subspace J. The equation ensures
        the transformation is valid within the subspace.

        Parameters
        ----------
        J : int
            Order of the spherical harmonics.
        a : float
            Rotation angle around the x-axis.
        b : float
            Rotation angle around the y-axis.
        c : float
            Rotation angle around the z-axis.

        Returns
        -------
        torch.Tensor
            A rank-deficient matrix for use in solving the Sylvester equation.

        Examples
        --------
        >>> _sylvester_submatrix(1, 1.0, 2.0, 3.0).shape
        torch.Size([...])
        """
        R_tensor = _R_tensor(a, b, c)
        R_irrep_J = irr_repr(J, a, b, c)
        return kron(R_tensor, torch.eye(R_irrep_J.size(0))) - \
                   kron(torch.eye(R_tensor.size(0)), R_irrep_J.t())

    random_angles = np.random.uniform(random_angle_lower,
                                      random_angle_higher,
                                      size=(num_samples, 3))

    null_space = get_matrices_kernel(
        [_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
    Q_J = null_space[0]
    Q_J = Q_J.view((2 * order_out + 1) * (2 * order_in + 1), 2 * J + 1)
    torch.set_default_dtype(original_dtype)

    return Q_J


def get_spherical_from_cartesian(cartesian: torch.Tensor,
                                 divide_radius_by: float = 1.0) -> torch.Tensor:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cartesian : torch.Tensor
        Cartesian coordinates tensor of shape [..., 3].
    divide_radius_by : float, optional
        Factor by which to divide the radius (default is 1.0).

    Returns
    -------
    torch.Tensor
        Spherical coordinates tensor of shape [..., 3] with [radius, azimuth (phi), elevation (theta)].

    Examples
    --------
    >>> from deepchem.utils.equivariance_utils import get_spherical_from_cartesian
    >>> cartesian = torch.tensor([[1.0, 1.0, 1.0]])
    >>> get_spherical_from_cartesian(cartesian)
    tensor([[1.7321, 0.7854, 0.9553]])

    >>> cartesian = torch.tensor([[0.0, 0.0, 1.0]])  # Point on Z-axis
    >>> get_spherical_from_cartesian(cartesian)
    tensor([[1.0000, 0.0000, 1.5708]])

    >>> cartesian = torch.tensor([[0.0, 0.0, -1.0]])  # Point on negative Z-axis
    >>> get_spherical_from_cartesian(cartesian)
    tensor([[1.0000, 3.1416, 1.5708]])
    """
    spherical = torch.zeros_like(cartesian)

    ind_radius = 0
    ind_alpha = 1
    ind_beta = 2

    cartesian_x = 2
    cartesian_y = 0
    cartesian_z = 1

    r_xy = cartesian[..., cartesian_x]**2 + cartesian[..., cartesian_y]**2

    spherical[..., ind_beta] = torch.atan2(torch.sqrt(r_xy),
                                           cartesian[..., cartesian_z])
    spherical[..., ind_alpha] = torch.atan2(cartesian[..., cartesian_y],
                                            cartesian[..., cartesian_x])

    if divide_radius_by == 1.0:
        spherical[..., ind_radius] = torch.sqrt(r_xy +
                                                cartesian[..., cartesian_z]**2)
    else:
        spherical[..., ind_radius] = torch.sqrt(
            r_xy + cartesian[..., cartesian_z]**2) / divide_radius_by

    return spherical


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kronecker product of two tensors needed to comput Q_J matrix.

    References:
    -----------
    - SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
    Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling
    NeurIPS 2020, https://arxiv.org/abs/2006.10503

    Parameters
    ----------
    a : torch.Tensor
        First input tensor of shape [*, m, n].
    b : torch.Tensor
        Second input tensor of shape [*, p, q].

    Returns
    -------
    torch.Tensor
        The Kronecker product of `a` and `b`.

    Examples
    --------
    >>> from deepchem.utils.equivariance_utils import kron
    >>> A = torch.tensor([[1, 2], [3, 4]])
    >>> B = torch.tensor([[0, 5], [6, 7]])
    >>> kron(A, B)
    tensor([[ 0,  5,  0, 10],
            [ 6,  7, 12, 14],
            [ 0, 15,  0, 20],
            [18, 21, 24, 28]])
    """
    if a.ndimension() == 2 and b.ndimension() == 2:
        return torch.einsum("ij,kl->ikjl", (a, b)).reshape(
            a.size(0) * b.size(0),
            a.size(1) * b.size(1))

    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def precompute_sh(r_ij: torch.Tensor, max_J: int) -> dict:
    """
    Precompute spherical harmonics up to a given order used in the forward pass of
    SE(3)-Transformer model.

    References:
    -----------
    - SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks
    Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, Max Welling
    NeurIPS 2020, https://arxiv.org/abs/2006.10503

    Parameters
    ----------
    r_ij : torch.Tensor
        Relative positions tensor.
    max_J : int
        Maximum order of the spherical harmonics.

    Returns
    -------
    dict
        A dictionary where each key corresponds to an order J and the value is a tensor
        of shape [B, N, K, 2J+1].

    Examples
    --------
    >>> from deepchem.utils.equivariance_utils import precompute_sh
    >>> r_ij = torch.tensor([[1.0, 0.5, 1.0]])  # Example spherical coordinates (radius, phi, theta)
    >>> precompute_sh(r_ij, max_J=2)
    {0: tensor([[0.2821]]), 1: tensor([[-0.1971, -0.2640, -0.3608]]), 2: tensor([[ 0.3255,  0.2381, -0.0392,  0.4359,  0.2090]])}
    """
    i_alpha = 1
    i_beta = 2

    Y_Js = {}
    sh = SphericalHarmonics()

    for J in range(max_J + 1):
        Y_Js[J] = sh.get(J,
                         theta=math.pi - r_ij[..., i_beta],
                         phi=r_ij[..., i_alpha],
                         refresh=False)

    sh.clear()
    return Y_Js


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
        q[k + m, k - abs(m)] = complex(-1j / 2**0.5)  # type: ignore

    # Set the diagonal elements for m = 0
    q[k, k] = 1

    # Construct the transformation matrix Q for m in range(1, k + 1)
    for m in range(1, k + 1):
        q[k + m, k + abs(m)] = (-1)**m / 2**0.5
        q[k + m, k - abs(m)] = complex(1j * (-1)**m / 2**0.5)  # type: ignore

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
    return q.to(
        dtype=dtype,
        device=device,
        copy=True,
        memory_format=torch.contiguous_format)  # type: ignore[call-overload]


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
        The Wigner D matrix of shape (#angles, 2k+1, 2k+1).

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
    tensor([[[ 0.8275,  0.1417,  0.5433],
             [ 0.0295,  0.9553, -0.2940],
             [-0.5607,  0.2593,  0.7863]],
    <BLANKLINE>
            [[ 0.7056,  0.2199,  0.6737],
             [ 0.0774,  0.9211, -0.3817],
             [-0.7044,  0.3214,  0.6329]]])
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
    D_matrix = torch.matrix_exp(gamma * (X[1].unsqueeze(0))) @ torch.matrix_exp(
        beta * (X[0].unsqueeze(0))) @ torch.matrix_exp(alpha *
                                                       (X[1].unsqueeze(0)))
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
