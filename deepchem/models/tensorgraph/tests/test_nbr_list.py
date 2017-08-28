import unittest

import numpy as np
import os
import tensorflow as tf
from nose.tools import assert_true
from tensorflow.python.framework import test_util

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.data.datasets import Databag
from deepchem.models.tensorgraph.layers import ReduceSum
from deepchem.models.tensorgraph.layers import Flatten
from deepchem.models.tensorgraph.layers import Feature, Label
from deepchem.models.tensorgraph.layers import Dense
from deepchem.models.tensorgraph.layers import ToFloat
from deepchem.models.tensorgraph.layers import Concat
from deepchem.models.tensorgraph.layers import NeighborList
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from deepchem.models.tensorgraph.layers import WeightedLinearCombo
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class TestNbrList(test_util.TensorFlowTestCase):
  """
  Test that tensorgraph neighbor-list works. 
  """

  def test_neighbor_list_simple(self):
    """Test that neighbor lists can be constructed."""
    N_atoms = 10
    start = 0
    stop = 12
    nbr_cutoff = 3
    ndim = 3
    M = 6
    X = np.random.rand(N_atoms, ndim)
    y = np.random.rand(N_atoms, 1)
    dataset = NumpyDataset(X, y)

    features = Feature(shape=(N_atoms, ndim))
    labels = Label(shape=(N_atoms,))
    nbr_list = NeighborList(
        N_atoms, M, ndim, nbr_cutoff, start, stop, in_layers=[features])
    nbr_list = ToFloat(in_layers=[nbr_list])
    # This isn't a meaningful loss, but just for test
    loss = ReduceSum(in_layers=[nbr_list])
    tg = dc.models.TensorGraph(use_queue=False)
    tg.add_output(nbr_list)
    tg.set_loss(loss)

    tg.build()

  def test_weighted_combo(self):
    """Tests that weighted linear combinations can be built"""
    N = 10
    n_features = 5

    X1 = NumpyDataset(np.random.rand(N, n_features))
    X2 = NumpyDataset(np.random.rand(N, n_features))
    y = NumpyDataset(np.random.rand(N))

    features_1 = Feature(shape=(None, n_features))
    features_2 = Feature(shape=(None, n_features))
    labels = Label(shape=(None,))

    combo = WeightedLinearCombo(in_layers=[features_1, features_2])
    out = ReduceSum(in_layers=[combo], axis=1)
    loss = ReduceSquareDifference(in_layers=[out, labels])

    databag = Databag({features_1: X1, features_2: X2, labels: y})

    tg = dc.models.TensorGraph(learning_rate=0.1, use_queue=False)
    tg.set_loss(loss)
    tg.fit_generator(databag.iterbatches(epochs=1))
    assert len(tg.get_layer_variables(combo)) >= 2
    assert len(tg.get_layer_variables(out)) == 0

  def test_neighbor_list_shape(self):
    """Test that NeighborList works."""
    N_atoms = 5
    start = 0
    stop = 12
    nbr_cutoff = 3
    ndim = 3
    M_nbrs = 2

    with self.test_session() as sess:
      coords = start + np.random.rand(N_atoms, ndim) * (stop - start)
      coords = tf.to_float(tf.stack(coords))
      nbr_list = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                              stop)(coords)
      nbr_list = nbr_list.eval()
      assert nbr_list.shape == (N_atoms, M_nbrs)

  def test_get_cells_1D(self):
    """Test neighbor-list method get_cells() in 1D"""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 1
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([1.0, 2.0, 8.0, 9.0])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      cells = nbr_list_layer.get_cells()
      cells_eval = cells.eval()
      true_cells = np.reshape(np.arange(10), (10, 1))
      np.testing.assert_array_almost_equal(cells_eval, true_cells)

  def test_get_closest_atoms_1D(self):
    """Test get_closest_atoms works correctly in 1D"""
    N_atoms = 4
    start = 0
    stop = 10
    n_cells = 10
    nbr_cutoff = 1
    ndim = 1
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([1.0, 2.0, 8.0, 9.0])
    coords = np.reshape(coords, (N_atoms, ndim))
    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      cells = nbr_list_layer.get_cells()
      closest_atoms = nbr_list_layer.get_closest_atoms(coords, cells)
      atoms_in_cells_eval = closest_atoms.eval()
      true_atoms_in_cells = np.reshape(
          np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3]), (n_cells, M_nbrs))
      np.testing.assert_array_almost_equal(atoms_in_cells_eval,
                                           true_atoms_in_cells)

  def test_get_neighbor_cells_1D(self):
    """Test that get_neighbor_cells works in 1D"""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 1
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([1.0, 2.0, 8.0, 9.0])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      cells = nbr_list_layer.get_cells()
      nbr_cells = nbr_list_layer.get_neighbor_cells(cells)
      nbr_cells_eval = nbr_cells.eval()
      true_nbr_cells = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 3], [3, 2, 4],
                                 [4, 3, 5], [5, 4, 6], [6, 5, 7], [7, 6, 8],
                                 [8, 7, 9], [9, 8, 7]])
      np.testing.assert_array_almost_equal(nbr_cells_eval, true_nbr_cells)

  def test_get_cells_for_atoms_1D(self):
    """Test that get_cells_for_atoms works in 1D"""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 1
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([1.0, 2.0, 8.0, 9.0])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      cells = nbr_list_layer.get_cells()
      cells_for_atoms = nbr_list_layer.get_cells_for_atoms(coords, cells)
      cells_for_atoms_eval = cells_for_atoms.eval()
      true_cells_for_atoms = np.array([[1], [2], [8], [9]])
      np.testing.assert_array_almost_equal(cells_for_atoms_eval,
                                           true_cells_for_atoms)

  def test_get_atoms_in_nbrs_1D(self):
    """Test get_atoms_in_brs in 1D"""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 1
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([1.0, 2.0, 8.0, 9.0])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      cells = nbr_list_layer.get_cells()
      uniques = nbr_list_layer.get_atoms_in_nbrs(coords, cells)

      uniques_eval = [unique.eval() for unique in uniques]
      uniques_eval = np.array(uniques_eval)

      true_uniques = np.array([[1], [0], [3], [2]])
      np.testing.assert_array_almost_equal(uniques_eval, true_uniques)

  def test_neighbor_list_1D(self):
    """Test neighbor list on 1D example."""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 1
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([1.0, 2.0, 8.0, 9.0])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      nbr_list = nbr_list_layer.compute_nbr_list(coords)
      nbr_list = np.squeeze(nbr_list.eval())
      np.testing.assert_array_almost_equal(nbr_list, np.array([1, 0, 3, 2]))

  def test_neighbor_list_2D(self):
    """Test neighbor list on 2D example."""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 2
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0]])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      nbr_list = nbr_list_layer.compute_nbr_list(coords)
      nbr_list = np.squeeze(nbr_list.eval())
      np.testing.assert_array_almost_equal(nbr_list, np.array([1, 0, 3, 2]))

  def test_neighbor_list_3D(self):
    """Test neighbor list on 3D example."""
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 3
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([[1.0, 0.0, 1.0], [2.0, 2.0, 2.0], [8.0, 8.0, 8.0],
                       [9.0, 9.0, 9.0]])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      nbr_list = nbr_list_layer.compute_nbr_list(coords)
      nbr_list = np.squeeze(nbr_list.eval())
      np.testing.assert_array_almost_equal(nbr_list, np.array([1, 0, 3, 2]))

  def test_neighbor_list_3D_empty_cells(self):
    """Test neighbor list on 3D example where cells are empty.

    Stresses the failure mode where the neighboring cells are empty
    so top_k will throw a failure.
    """
    N_atoms = 4
    start = 0
    stop = 10
    nbr_cutoff = 1
    ndim = 3
    M_nbrs = 1
    # 1 and 2 are nbrs. 8 and 9 are nbrs
    coords = np.array([[1.0, 0.0, 1.0], [2.0, 5.0, 2.0], [8.0, 8.0, 8.0],
                       [9.0, 9.0, 9.0]])
    coords = np.reshape(coords, (N_atoms, ndim))

    with self.test_session() as sess:
      coords = tf.convert_to_tensor(coords, dtype=tf.float32)
      nbr_list_layer = NeighborList(N_atoms, M_nbrs, ndim, nbr_cutoff, start,
                                    stop)
      nbr_list = nbr_list_layer.compute_nbr_list(coords)
      nbr_list = np.squeeze(nbr_list.eval())
      np.testing.assert_array_almost_equal(nbr_list, np.array([-1, -1, 3, 2]))

  def test_neighbor_list_vina(self):
    """Test under conditions closer to Vina usage."""
    N_atoms = 5
    M_nbrs = 2
    ndim = 3
    start = 0
    stop = 4
    nbr_cutoff = 1

    X = NumpyDataset(start + np.random.rand(N_atoms, ndim) * (stop - start))

    coords = Feature(shape=(N_atoms, ndim))

    # Now an (N, M) shape
    nbr_list = NeighborList(
        N_atoms, M_nbrs, ndim, nbr_cutoff, start, stop, in_layers=[coords])

    nbr_list = ToFloat(in_layers=[nbr_list])
    flattened = Flatten(in_layers=[nbr_list])
    dense = Dense(out_channels=1, in_layers=[flattened])
    output = ReduceSum(in_layers=[dense])

    tg = dc.models.TensorGraph(learning_rate=0.1, use_queue=False)
    tg.set_loss(output)

    databag = Databag({coords: X})
    tg.fit_generator(databag.iterbatches(epochs=1))

  # TODO(rbharath): Move this into a model directly
  #def test_vina(self):
  #  """Test that vina graph can be constructed in TensorGraph."""
  #  N_protein = 4
  #  N_ligand = 1
  #  N_atoms = 5
  #  M_nbrs = 2
  #  ndim = 3
  #  start = 0
  #  stop = 4
  #  nbr_cutoff = 1

  #  X_prot = NumpyDataset(start + np.random.rand(N_protein, ndim) * (stop -
  #                                                                   start))
  #  X_ligand = NumpyDataset(start + np.random.rand(N_ligand, ndim) * (stop -
  #                                                                    start))
  #  y = NumpyDataset(np.random.rand(
  #      1,))

  #  # TODO(rbharath): Mysteriously, the actual atom types aren't
  #  # used in the current implementation. This is obviously wrong, but need
  #  # to dig out why this is happening.
  #  prot_coords = Feature(shape=(N_protein, ndim))
  #  ligand_coords = Feature(shape=(N_ligand, ndim))
  #  labels = Label(shape=(1,))

  #  coords = Concat(in_layers=[prot_coords, ligand_coords], axis=0)

  #  #prot_Z = Feature(shape=(N_protein,), dtype=tf.int32)
  #  #ligand_Z = Feature(shape=(N_ligand,), dtype=tf.int32)
  #  #Z = Concat(in_layers=[prot_Z, ligand_Z], axis=0)

  #  # Now an (N, M) shape
  #  nbr_list = NeighborList(
  #      N_protein + N_ligand,
  #      M_nbrs,
  #      ndim,
  #      nbr_cutoff,
  #      start,
  #      stop,
  #      in_layers=[coords])

  #  # Shape (N, M)
  #  dists = InteratomicL2Distances(
  #      N_protein + N_ligand, M_nbrs, ndim, in_layers=[coords, nbr_list])

  #  repulsion = VinaRepulsion(in_layers=[dists])
  #  hydrophobic = VinaHydrophobic(in_layers=[dists])
  #  hbond = VinaHydrogenBond(in_layers=[dists])
  #  gauss_1 = VinaGaussianFirst(in_layers=[dists])
  #  gauss_2 = VinaGaussianSecond(in_layers=[dists])

  #  # Shape (N, M)
  #  interactions = WeightedLinearCombo(
  #      in_layers=[repulsion, hydrophobic, hbond, gauss_1, gauss_2])

  #  # Shape (N, M)
  #  thresholded = Cutoff(in_layers=[dists, interactions])

  #  # Shape (N, M)
  #  free_energies = VinaNonlinearity(in_layers=[thresholded])
  #  free_energy = ReduceSum(in_layers=[free_energies])

  #  loss = L2Loss(in_layers=[free_energy, labels])

  #  databag = Databag({prot_coords: X_prot, ligand_coords: X_ligand, labels: y})

  #  tg = dc.models.TensorGraph(learning_rate=0.1, use_queue=False)
  #  tg.set_loss(loss)
  #  tg.fit_generator(databag.iterbatches(epochs=1))
