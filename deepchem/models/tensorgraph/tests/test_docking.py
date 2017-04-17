import unittest

import numpy as np
import os
from nose.tools import assert_true

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.data.datasets import Databag
from deepchem.models.tensorgraph.layers import ReduceSum 
from deepchem.models.tensorgraph.layers import Feature, Label
from deepchem.models.tensorgraph.layers import ToFloat
from deepchem.models.tensorgraph.layers import NeighborList
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from deepchem.models.tensorgraph.layers import WeightedLinearCombo
from deepchem.models.tensorgraph.layers import InteratomicL2Distances
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class TestDocking(unittest.TestCase):
  """
  Test that tensorgraph docking-style models work. 
  """

  def test_neighbor_list(self):
    """Test that neighbor lists can be constructed."""
    N_atoms = 10
    start = 0
    stop = 12
    nbr_cutoff = 3
    ndim = 3
    M = 6
    k = 5
    # The number of cells which we should theoretically have
    n_cells = int(((stop - start) / nbr_cutoff)**ndim)

    X = np.random.rand(N_atoms, ndim)
    y = np.random.rand(N_atoms, 1)
    dataset = NumpyDataset(X, y)

    features = Feature(shape=(N_atoms, ndim))
    labels = Label(shape=(N_atoms,))
    nbr_list = NeighborList(N_atoms, M, ndim, n_cells, k, nbr_cutoff,
                            in_layers=[features])
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

  def test_vina(self):
    """Test that vina graph can be constructed in TensorGraph."""

    prot_coords = Features(shape=(N_protein, 3))
    prot_Z = Features(shape=(N_protein,), dtype=tf.int32)
    ligand_coords = Features(shape=(N_ligand, 3))
    ligand_Z = Features(shape=(N_ligand,), dtype=tf.int32)
    labels = Label(shape=(1,))

    coords = Concat(in_layers=[prot_coords, ligand_coords], axis=0)
    Z = Concat(in_layers=[prot_Z, ligand_Z], axis=0)

    # Now an (N, M) shape
    nbr_list = NeighborList(N_protein+N_ligand, M, ndim, n_cells, k,
                            nbr_cutoff, in_layers=[coords])

  def test_interatomic_distances(self):
    """Test that the interatomic distance calculation works."""
    N_atoms = 5
    M = 2
    ndim = 3

    coords = np.random.rand(N_atoms, ndim)
    nbr_list = np.random.randint(0, N_atoms, size=(N_atoms, M))

    coords_tensor = tf.convert_to_tensor(coords)
    nbr_list_tensor = tf.convert_to_tensor(nbr_list)

    dist_tensor = 

