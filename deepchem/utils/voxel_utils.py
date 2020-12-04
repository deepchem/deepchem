"""
Various utilities around voxel grids.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def convert_atom_to_voxel(coordinates: np.ndarray, atom_index: int,
                          box_width: float, voxel_width: float) -> np.ndarray:
  """Converts atom coordinates to an i,j,k grid index.

  This function offsets molecular atom coordinates by
  (box_width/2, box_width/2, box_width/2) and then divides by
  voxel_width to compute the voxel indices.

  Parameters
  -----------
  coordinates: np.ndarray
    Array with coordinates of all atoms in the molecule, shape (N, 3).
  atom_index: int
    Index of an atom in the molecule.
  box_width: float
    Size of the box in Angstroms.
  voxel_width: float
    Size of a voxel in Angstroms

  Returns
  -------
  indices: np.ndarray
    A 1D numpy array of length 3 with `[i, j, k]`, the voxel coordinates
    of specified atom.
  """

  indices = np.floor(
      (coordinates[atom_index] + box_width / 2.0) / voxel_width).astype(int)

  if ((indices < 0) | (indices >= box_width / voxel_width)).any():
    logger.warning('Coordinates are outside of the box (atom id = %s,'
                   ' coords xyz = %s, coords in box = %s' %
                   (atom_index, coordinates[atom_index], indices))
  return indices


def convert_atom_pair_to_voxel(coordinates_tuple: Tuple[np.ndarray, np.ndarray],
                               atom_index_pair: Tuple[int, int],
                               box_width: float,
                               voxel_width: float) -> np.ndarray:
  """Converts a pair of atoms to i,j,k grid indexes.


  Parameters
  ----------
  coordinates_tuple: Tuple[np.ndarray, np.ndarray]
    A tuple containing two molecular coordinate arrays of shapes `(N, 3)` and `(M, 3)`.
  atom_index_pair: Tuple[int, int]
    A tuple of indices for the atoms in the two molecules.
  box_width: float
    Size of the box in Angstroms.
  voxel_width: float
    Size of a voxel in Angstroms

  Returns
  -------
  indices_list: np.ndarray
    A numpy array of shape `(2, 3)`, where `3` is `[i, j, k]` of the
    voxel coordinates of specified atom.
  """

  indices_list = []
  for coordinates, atom_index in zip(coordinates_tuple, atom_index_pair):
    indices_list.append(
        convert_atom_to_voxel(coordinates, atom_index, box_width, voxel_width))
  return np.array(indices_list)


def voxelize(get_voxels: Callable[..., Any],
             coordinates: np.ndarray,
             box_width: float = 16.0,
             voxel_width: float = 1.0,
             hash_function: Optional[Callable[..., Any]] = None,
             feature_dict: Optional[Dict[Any, Any]] = None,
             feature_list: Optional[List[Union[int, Tuple[int]]]] = None,
             nb_channel: int = 16,
             dtype: str = 'int') -> np.ndarray:
  """Helper function to voxelize inputs.

  This helper function helps convert a hash function which
  specifies spatial features of a molecular complex into a voxel
  tensor. This utility is used by various featurizers that generate
  voxel grids.

  Parameters
  ----------
  get_voxels: Function
    Function that voxelizes inputs
  coordinates: np.ndarray
    Contains the 3D coordinates of a molecular system.
  box_width: float, optional (default 16.0)
    Size of a box in which voxel features are calculated. Box
    is centered on a ligand centroid.
  voxel_width: float, optional (default 1.0)
    Size of a 3D voxel in a grid in Angstroms.
  hash_function: Function
    Used to map feature choices to voxel channels.
  feature_dict: Dict, optional (default None)
    Keys are atom indices or tuples of atom indices, the values are
    computed features. If `hash_function is not None`, then the values
    are hashed using the hash function into `[0, nb_channels)` and
    this channel at the voxel for the given key is incremented by `1`
    for each dictionary entry. If `hash_function is None`, then the
    value must be a vector of size `(n_channels,)` which is added to
    the existing channel values at that voxel grid.
  feature_list: List, optional (default None)
    List of atom indices or tuples of atom indices. This can only be
    used if `nb_channel==1`. Increments the voxels corresponding to
    these indices by `1` for each entry.
  nb_channel: int, , optional (default 16)
    The number of feature channels computed per voxel. Should
    be a power of 2.
  dtype: str ('int' or 'float'), optional (default 'int')
    The type of the numpy ndarray created to hold features.

  Returns
  -------
  feature_tensor: np.ndarray
    The voxel of the input with the shape
    `(voxels_per_edge, voxels_per_edge, voxels_per_edge, nb_channel)`.
  """
  # Number of voxels per one edge of box to voxelize.
  voxels_per_edge = int(box_width / voxel_width)
  if dtype == "int":
    feature_tensor = np.zeros(
        (voxels_per_edge, voxels_per_edge, voxels_per_edge, nb_channel),
        dtype=np.int8)
  else:
    feature_tensor = np.zeros(
        (voxels_per_edge, voxels_per_edge, voxels_per_edge, nb_channel),
        dtype=np.float16)
  if feature_dict is not None:
    for key, features in feature_dict.items():
      voxels = get_voxels(coordinates, key, box_width, voxel_width)
      if len(voxels.shape) == 1:
        voxels = np.expand_dims(voxels, axis=0)
      for voxel in voxels:
        if ((voxel >= 0) & (voxel < voxels_per_edge)).all():
          if hash_function is not None:
            feature_tensor[voxel[0], voxel[1], voxel[2],
                           hash_function(features, nb_channel)] += 1.0
          else:
            feature_tensor[voxel[0], voxel[1], voxel[2], 0] += features
  elif feature_list is not None:
    for key in feature_list:
      voxels = get_voxels(coordinates, key, box_width, voxel_width)
      for voxel in voxels:
        if ((voxel >= 0) & (voxel < voxels_per_edge)).all():
          feature_tensor[voxel[0], voxel[1], voxel[2], 0] += 1.0

  return feature_tensor
