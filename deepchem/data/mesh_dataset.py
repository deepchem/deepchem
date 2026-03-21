"""
Dataset abstraction for finite element mesh problems.

Mirrors DeepChem's Dataset interface closely so porting to
deepchem.data.MeshDataset in the final PR requires zero algorithmic changes.

Each sample represents one FEM problem instance:
  - nodes          : (N, 2)  node coordinates
  - elements       : (E, 3)  triangle connectivity
  - boundary_mask  : (N,)    True at Dirichlet boundary nodes
  - boundary_values: (N,)    known u values at boundary nodes
  - source         : (N,)    source term f(x,y) at nodes
  - solution       : (N,)    ground-truth u  [optional, for inverse problems]
"""

from typing import Dict, Iterator, List, Optional
import numpy as np
import torch


class MeshDataset:
    """Dataset of triangular finite element mesh problems.

    Designed to work directly with FEMSolver and MeshFeaturizer.
    Mirrors DeepChem's Dataset API (len, getitem, iter) so porting
    to deepchem.data.MeshDataset in the GSoC PR is a drop-in.

    Parameters
    ----------
    nodes : list of np.ndarray, each shape (N_i, 2)
        Node (x, y) coordinates per sample.
    elements : list of np.ndarray, each shape (E_i, 3), dtype int
        Triangle connectivity (node indices) per sample.
    boundary_masks : list of np.ndarray, each shape (N_i,), dtype bool
        True at Dirichlet boundary nodes.
    boundary_values : list of np.ndarray, each shape (N_i,)
        Known u values at boundary nodes.
    sources : list of np.ndarray, each shape (N_i,)
        Source term f(x,y) evaluated at nodes.
    solutions : list of np.ndarray or None, each shape (N_i,)
        Ground-truth u values. Required for inverse problem training.

    Example
    -------
    >>> import numpy as np
    >>> from mesh_dataset import MeshDataset
    >>> nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float32)
    >>> elements = np.array([[0,1,2],[1,3,2]], dtype=np.int64)
    >>> bm = np.array([True, True, True, False])
    >>> bv = np.zeros(4, dtype=np.float32)
    >>> src = np.ones(4, dtype=np.float32)
    >>> ds = MeshDataset([nodes], [elements], [bm], [bv], [src])
    >>> len(ds)
    1
    >>> sample = ds[0]
    >>> sample['nodes'].shape
    torch.Size([4, 2])
    """

    def __init__(
        self,
        nodes: List[np.ndarray],
        elements: List[np.ndarray],
        boundary_masks: List[np.ndarray],
        boundary_values: List[np.ndarray],
        sources: List[np.ndarray],
        solutions: Optional[List[np.ndarray]] = None,
    ):
        lengths = [
            len(nodes), len(elements), len(boundary_masks),
            len(boundary_values), len(sources)
        ]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All input lists must have equal length. Got: {lengths}"
            )
        if solutions is not None and len(solutions) != len(nodes):
            raise ValueError(
                f"solutions length {len(solutions)} != nodes length {len(nodes)}"
            )

        self.nodes = nodes
        self.elements = elements
        self.boundary_masks = boundary_masks
        self.boundary_values = boundary_values
        self.sources = sources
        self.solutions = solutions

    # core interface

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return sample at idx as a dict of torch Tensors.

        Returns
        -------
        dict with keys:
            nodes            : FloatTensor (N, 2)
            elements         : LongTensor  (E, 3)
            boundary_mask    : BoolTensor  (N,)
            boundary_values  : FloatTensor (N,)
            source           : FloatTensor (N,)
            solution         : FloatTensor (N,)  — only if solutions provided
        """
        sample = {
            'nodes': torch.tensor(
                self.nodes[idx], dtype=torch.float32),
            'elements': torch.tensor(
                self.elements[idx], dtype=torch.long),
            'boundary_mask': torch.tensor(
                self.boundary_masks[idx], dtype=torch.bool),
            'boundary_values': torch.tensor(
                self.boundary_values[idx], dtype=torch.float32),
            'source': torch.tensor(
                self.sources[idx], dtype=torch.float32),
        }
        if self.solutions is not None:
            sample['solution'] = torch.tensor(
                self.solutions[idx], dtype=torch.float32)
        return sample

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(len(self)):
            yield self[i]

    # utilities

    def stats(self) -> Dict:
        """Return summary statistics about the dataset.

        Returns
        -------
        dict with keys:
            n_samples      : int
            mean_nodes     : float — average nodes per mesh
            mean_elements  : float — average elements per mesh
            has_solutions  : bool
        """
        return {
            'n_samples': len(self),
            'mean_nodes': float(
                np.mean([n.shape[0] for n in self.nodes])),
            'mean_elements': float(
                np.mean([e.shape[0] for e in self.elements])),
            'has_solutions': self.solutions is not None,
        }

    @classmethod
    def from_featurizer(
        cls,
        featurizer,
        source_fns,
        boundary_fns=None,
        solution_fns=None,
    ) -> 'MeshDataset':
        """Construct a MeshDataset from a MeshFeaturizer.

        Parameters
        ----------
        featurizer : MeshFeaturizer
        source_fns : list of callable  f(x, y) -> array
        boundary_fns : list of callable or None  g(x, y) -> array
        solution_fns : list of callable or None  u(x, y) -> array
            If provided, exact solutions are stored for validation.

        Returns
        -------
        MeshDataset
        """
        meshes = featurizer.featurize(source_fns, boundary_fns)
        solutions = None
        if solution_fns is not None:
            solutions = []
            for mesh, sol_fn in zip(meshes, solution_fns):
                x = mesh['nodes'][:, 0]
                y = mesh['nodes'][:, 1]
                solutions.append(sol_fn(x, y).astype(np.float32))

        return cls(
            nodes=[m['nodes'] for m in meshes],
            elements=[m['elements'] for m in meshes],
            boundary_masks=[m['boundary_mask'] for m in meshes],
            boundary_values=[m['boundary_values'] for m in meshes],
            sources=[m['source'] for m in meshes],
            solutions=solutions,
        )