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
from deepchem.models.tensorgraph.layers import Feature, Label
from deepchem.models.tensorgraph.layers import ToFloat
from deepchem.models.tensorgraph.layers import Concat
from deepchem.models.tensorgraph.layers import NeighborList
from deepchem.models.tensorgraph.layers import ReduceSquareDifference
from deepchem.models.tensorgraph.layers import WeightedLinearCombo
from deepchem.models.tensorgraph.layers import InteratomicL2Distances
from deepchem.models.tensorgraph.layers import Cutoff
from deepchem.models.tensorgraph.layers import VinaRepulsion
from deepchem.models.tensorgraph.layers import VinaNonlinearity
from deepchem.models.tensorgraph.layers import VinaHydrophobic
from deepchem.models.tensorgraph.layers import VinaHydrogenBond
from deepchem.models.tensorgraph.layers import VinaGaussianFirst
from deepchem.models.tensorgraph.layers import VinaGaussianSecond
from deepchem.models.tensorgraph.layers import L2LossLayer
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class TestDocking(test_util.TensorFlowTestCase):
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
    N_protein = 4
    N_ligand = 1
    N_atoms = 5
    M_nbrs = 2
    ndim = 3
    k = 5
    start = 0
    stop = 4
    nbr_cutoff = 1
    # The number of cells which we should theoretically have
    n_cells = ((stop - start) / nbr_cutoff)**ndim

    X_prot = NumpyDataset(np.random.rand(N_protein, ndim))
    X_ligand = NumpyDataset(np.random.rand(N_ligand, ndim))
    y = NumpyDataset(np.random.rand(1,))

    # TODO(rbharath): Mysteriously, the actual atom types aren't
    # used in the current implementation. This is obviously wrong, but need
    # to dig out why this is happening.
    prot_coords = Feature(shape=(N_protein, ndim))
    ligand_coords = Feature(shape=(N_ligand, ndim))
    labels = Label(shape=(1,))

    coords = Concat(in_layers=[prot_coords, ligand_coords], axis=0)

    #prot_Z = Feature(shape=(N_protein,), dtype=tf.int32)
    #ligand_Z = Feature(shape=(N_ligand,), dtype=tf.int32)
    #Z = Concat(in_layers=[prot_Z, ligand_Z], axis=0)

    # Now an (N, M) shape
    nbr_list = NeighborList(N_protein+N_ligand, M_nbrs, ndim, n_cells, k,
                            nbr_cutoff, in_layers=[coords])

    # Shape (N, M)
    dists = InteratomicL2Distances(N_protein+N_ligand, M_nbrs, ndim,
                                   in_layers=[coords, nbr_list])

    repulsion = VinaRepulsion(in_layers=[dists])
    hydrophobic = VinaHydrophobic(in_layers=[dists])
    hbond = VinaHydrogenBond(in_layers=[dists])
    gauss_1 = VinaGaussianFirst(in_layers=[dists]) 
    gauss_2 = VinaGaussianSecond(in_layers=[dists]) 

    # Shape (N, M)
    interactions = WeightedLinearCombo(
        in_layers=[repulsion, hydrophobic, hbond, gauss_1, gauss_2])
    
    # Shape (N, M)
    thresholded = Cutoff(in_layers=[dists, interactions])

    # Shape (N, M)
    free_energies = VinaNonlinearity(in_layers=[thresholded])
    free_energy = ReduceSum(in_layers=[free_energies])
    
    loss = L2LossLayer(in_layers=[free_energy, labels])
    
    databag = Databag({prot_coords: X_prot, ligand_coords: X_ligand,
                       labels: y})

    tg = dc.models.TensorGraph(learning_rate=0.1, use_queue=False)
    tg.set_loss(loss)
    tg.fit_generator(databag.iterbatches(epochs=1))
    
    

  def test_interatomic_distances(self):
    """Test that the interatomic distance calculation works."""
    N_atoms = 5
    M_nbrs = 2
    ndim = 3

    with self.test_session() as sess:
      coords = np.random.rand(N_atoms, ndim)
      nbr_list = np.random.randint(0, N_atoms, size=(N_atoms, M_nbrs))

      coords_tensor = tf.convert_to_tensor(coords)
      nbr_list_tensor = tf.convert_to_tensor(nbr_list)

      dist_tensor = InteratomicL2Distances(N_atoms, M_nbrs, ndim)(
          coords_tensor, nbr_list_tensor)

      dists = dist_tensor.eval()
      assert dists.shape == (N_atoms, M_nbrs)
