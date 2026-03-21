"""
Featurizer that converts PDE problem specifications into structured mesh dicts
for use with MeshDataset and FEMSolver.

Given source and boundary functions, generates a uniform triangular mesh on
[0,1]² and evaluates the functions at mesh nodes. Designed to mirror
DeepChem's Featurizer interface for easy porting.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np


def _make_unit_square_mesh(
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a uniform triangular mesh on [0,1]².

    Parameters
    ----------
    nx, ny : int
        Number of divisions along x and y axes.

    Returns
    -------
    nodes : np.ndarray (N, 2), float32
    elements : np.ndarray (2*nx*ny, 3), int64
    boundary_mask : np.ndarray (N,), bool
    """
    xs = np.linspace(0.0, 1.0, nx + 1, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elems = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = j * (nx + 1) + i + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = (j + 1) * (nx + 1) + i + 1
            elems.append([n00, n10, n01])
            elems.append([n10, n11, n01])

    elements = np.array(elems, dtype=np.int64)
    boundary_mask = (
        (nodes[:, 0] == 0.0) | (nodes[:, 0] == 1.0) |
        (nodes[:, 1] == 0.0) | (nodes[:, 1] == 1.0)
    )
    return nodes, elements, boundary_mask


class MeshFeaturizer:
    """Generates structured triangular mesh inputs for FEM problems.

    Given callables for the source term f(x,y) and boundary values g(x,y),
    produces mesh dicts ready for MeshDataset. Mirrors DeepChem's Featurizer
    interface for drop-in porting.

    Parameters
    ----------
    nx : int, default 8
        Mesh divisions along x. More divisions = finer mesh = higher accuracy.
    ny : int, default 8
        Mesh divisions along y.

    Example
    -------
    >>> from mesh_featurizer import MeshFeaturizer
    >>> import numpy as np
    >>> feat = MeshFeaturizer(nx=4, ny=4)
    >>> meshes = feat.featurize(
    ...     source_fns=[lambda x, y: np.ones_like(x)],
    ...     boundary_fns=[lambda x, y: np.zeros_like(x)])
    >>> meshes[0]['nodes'].shape
    (25, 2)
    >>> meshes[0]['elements'].shape
    (32, 3)
    """

    def __init__(self, nx: int = 8, ny: int = 8):
        self.nx = nx
        self.ny = ny
        self._nodes, self._elements, self._boundary_mask = (
            _make_unit_square_mesh(nx, ny)
        )

    @property
    def n_nodes(self) -> int:
        """Total number of nodes in the generated mesh."""
        return self._nodes.shape[0]

    @property
    def n_elements(self) -> int:
        """Total number of triangular elements in the generated mesh."""
        return self._elements.shape[0]

    def featurize(
        self,
        source_fns: List[Callable],
        boundary_fns: Optional[List[Callable]] = None,
    ) -> List[dict]:
        """Featurize a list of PDE problems into mesh dicts.

        Parameters
        ----------
        source_fns : list of callable
            Each callable takes arrays (x, y) of node coordinates and
            returns the source term f as a 1D float32 array of shape (N,).
        boundary_fns : list of callable or None
            Each callable takes (x, y) and returns boundary values g as
            a 1D float32 array of shape (N,).
            If None, zero Dirichlet BCs are used for all samples.

        Returns
        -------
        list of dict, one per problem, with keys:
            nodes           : np.ndarray (N, 2), float32
            elements        : np.ndarray (E, 3), int64
            boundary_mask   : np.ndarray (N,),   bool
            boundary_values : np.ndarray (N,),   float32
            source          : np.ndarray (N,),   float32
        """
        if boundary_fns is None:
            boundary_fns = [
                lambda x, y: np.zeros_like(x)
            ] * len(source_fns)

        if len(source_fns) != len(boundary_fns):
            raise ValueError(
                f"source_fns ({len(source_fns)}) and boundary_fns "
                f"({len(boundary_fns)}) must have equal length."
            )

        x = self._nodes[:, 0]
        y = self._nodes[:, 1]
        results = []

        for src_fn, bc_fn in zip(source_fns, boundary_fns):
            source = src_fn(x, y).astype(np.float32)
            boundary_values = bc_fn(x, y).astype(np.float32)

            if source.shape != (self.n_nodes,):
                raise ValueError(
                    f"source_fn must return array of shape ({self.n_nodes},), "
                    f"got {source.shape}"
                )

            results.append({
                'nodes': self._nodes.copy(),
                'elements': self._elements.copy(),
                'boundary_mask': self._boundary_mask.copy(),
                'boundary_values': boundary_values,
                'source': source,
            })

        return results

    def refine(self, factor: int = 2) -> 'MeshFeaturizer':
        """Return a new MeshFeaturizer with a finer mesh.

        Parameters
        ----------
        factor : int
            Multiply nx and ny by this factor.

        Returns
        -------
        MeshFeaturizer with nx*factor, ny*factor divisions.
        """
        return MeshFeaturizer(nx=self.nx * factor, ny=self.ny * factor)