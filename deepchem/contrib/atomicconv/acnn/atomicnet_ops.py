__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

"""Tensorflow Ops for Atomicnet."""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

### Atomicnet coordinate transform ops ###

def gather_neighbors(X, nbr_indices, B, N, M, d):
  """Gathers the neighbor subsets of the atoms in X.

  B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

  Parameters
  ----------
  X: tf.Tensor of shape (B, N, d)
    Coordinates/features tensor.
  atom_indices: tf.Tensor of shape (B, M)
    Neighbor list for single atom.

  Returns
  -------
  neighbors: tf.Tensor of shape (B, M, d)
    Neighbor coordinates/features tensor for single atom.

  """

  example_tensors = tf.unstack(X, axis=0)
  example_nbrs = tf.unstack(nbr_indices, axis=0)
  all_nbr_coords = []
  for example, (example_tensor,
                example_nbr) in enumerate(zip(example_tensors, example_nbrs)):
    nbr_coords = tf.gather(example_tensor, example_nbr)
    all_nbr_coords.append(nbr_coords)
  neighbors = tf.stack(all_nbr_coords)
  return neighbors


def DistanceTensor(X, Nbrs, boxsize, B, N, M, d):
  """Calculates distance tensor for batch of molecules.

  B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

  Parameters
  ----------
  X: tf.Tensor of shape (B, N, d)
    Coordinates/features tensor.
  Nbrs: tf.Tensor of shape (B, N, M)
    Neighbor list tensor.
  boxsize: float or None
    Simulation box length [Angstrom].

  Returns
  -------
  D: tf.Tensor of shape (B, N, M, d)
    Coordinates/features distance tensor.

  """

  atom_tensors = tf.unstack(X, axis=1)
  nbr_tensors = tf.unstack(Nbrs, axis=1)
  D = []
  if boxsize is not None:
    for atom, atom_tensor in enumerate(atom_tensors):
      nbrs = gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
      nbrs_tensors = tf.unstack(nbrs, axis=1)
      for nbr, nbr_tensor in enumerate(nbrs_tensors):
        _D = tf.subtract(nbr_tensor, atom_tensor)
        _D = tf.subtract(_D, boxsize * tf.round(tf.div(_D, boxsize)))
        D.append(_D)
  else:
    for atom, atom_tensor in enumerate(atom_tensors):
      nbrs = gather_neighbors(X, nbr_tensors[atom], B, N, M, d)
      nbrs_tensors = tf.unstack(nbrs, axis=1)
      for nbr, nbr_tensor in enumerate(nbrs_tensors):
        _D = tf.subtract(nbr_tensor, atom_tensor)
        D.append(_D)
  D = tf.stack(D)
  D = tf.transpose(D, perm=[1, 0, 2])
  D = tf.reshape(D, [B, N, M, d])
  return D


def DistanceMatrix(D):
  """Calcuates the distance matrix from the distance tensor

  B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_features

  Parameters
  ----------
  D: tf.Tensor of shape (B, N, M, d)
    Distance tensor.

  Returns
  -------
  R: tf.Tensor of shape (B, N, M)
     Distance matrix.

  """

  R = tf.reduce_sum(tf.multiply(D, D), 3)
  R = tf.sqrt(R)
  return R


### Atomicnet symmetry function kernel ops ###


def GaussianDistanceMatrix(R, rs, e):
  """Calculates gaussian distance matrix.

  B = batch_size, N = max_num_atoms, M = max_num_neighbors

  Parameters
  ----------
    R [B, N, M]: tf.Tensor
      Distance matrix.
    rs: tf.Variable
      Gaussian distance matrix mean.
    e: tf.Variable
      Gaussian distance matrix width (e = .5/std**2).

  Returns
  -------
    retval [B, N, M]: tf.Tensor
      Gaussian distance matrix.

  """

  return tf.exp(-e * (R - rs)**2)


def RadialCutoff(R, rc):
  """Calculates radial cutoff matrix.

  B = batch_size, N = max_num_atoms, M = max_num_neighbors

  Parameters
  ----------
    R [B, N, M]: tf.Tensor
      Distance matrix.
    rc: tf.Variable
      Interaction cutoff [Angstrom].

  Returns
  -------
    FC [B, N, M]: tf.Tensor
      Radial cutoff matrix.

  """

  T = 0.5 * (tf.cos(np.pi * R / (rc)) + 1)
  E = tf.zeros_like(T)
  cond = tf.less_equal(R, rc)
  FC = tf.where(cond, T, E)
  return FC


### Atomicnet symmetry function ops ###


def RadialSymmetryFunction(R, rc, rs, e):
  """Calculates radial symmetry function.

  B = batch_size, N = max_num_atoms, M = max_num_neighbors, d = num_filters

  Parameters
  ----------
  R: tf.Tensor of shape (B, N, M)
    Distance matrix.
  rc: float
    Interaction cutoff [Angstrom].
  rs: float
    Gaussian distance matrix mean.
  e: float
    Gaussian distance matrix width.

  Returns
  -------
  retval: tf.Tensor of shape (B, N, M)
    Radial symmetry function (before summation)

  """

  with tf.name_scope(None, "NbrRadialSymmetryFunction", [rc, rs, e]):
    rc = tf.get_variable('rc', initializer=tf.constant(rc))
    rs = tf.get_variable('rs', initializer=tf.constant(rs))
    e = tf.get_variable('e', initializer=tf.constant(e))
    K = GaussianDistanceMatrix(R, rs, e)
    FC = RadialCutoff(R, rc)
  return tf.multiply(K, FC)


### Atomcnet symmetry function layer ops ###


def AtomicConvolutionLayer(X, Nbrs, Nbrs_Z, atom_types, radial_params, boxsize,
                           B, N, M, d):
  """Atomic convoluation layer

  N = max_num_atoms, M = max_num_neighbors, B = batch_size, d = num_features
  l = num_radial_filters * num_atom_types

  Parameters
  ----------
  X: tf.Tensor of shape (B, N, d)
    Coordinates/features.
  Nbrs: tf.Tensor of shape (B, N, M)
    Neighbor list.
  Nbrs_Z: tf.Tensor of shape (B, N, M)
    Atomic numbers of neighbor atoms.
  atom_types: list or None
    Of length a, where a is number of atom types for filtering.
  radial_params: list
    Of length l, where l is number of radial filters learned.
  boxsize: float or None
    Simulation box length [Angstrom].
  N: int
    Maximum number of atoms
  M: int
    Maximum number of neighbors
  d: int
    Number of coordinates/features/filters

  Returns
  -------
  layer: tf.Tensor of shape (l, B, N)
    A new tensor representing the output of the atomic conv layer 

  """

  D = DistanceTensor(X, Nbrs, boxsize, B, N, M, d)
  R = DistanceMatrix(D)
  sym = []
  rsf_zeros = tf.zeros((B, N, M))
  for ix, param in enumerate(radial_params):
    with tf.variable_scope('radial_{}'.format(ix)):
        # We apply the radial pooling filter before atom type conv
        # to reduce computation
        rsf = RadialSymmetryFunction(R, *param)

    if not atom_types:
      cond = tf.not_equal(Nbrs_Z, 0.0)
      sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))
    else:
      for j in range(len(atom_types)):
        cond = tf.equal(Nbrs_Z, atom_types[j])
        sym.append(tf.reduce_sum(tf.where(cond, rsf, rsf_zeros), 2))

  # Pack l (B, N) tensors into one (l, B, N) tensor
  # Transpose to (B, N, l) for conv layer stacking
  # done inside conv_layer loops to reduce transpose ops
  # Final layer should be shape (N, B, l) to pass into tf.map_fn
  layer = tf.stack(sym)
  return layer


### Misc convenience ops ###

def create_symmetry_parameters(radial):
  rp = []
  for r0 in radial[0]:
    for r1 in radial[1]:
      for r2 in radial[2]:
        rp.append([r0, r1, r2])
  return rp
