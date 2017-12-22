"""
Testing construction of Vina models. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import tensorflow as tf
import deepchem as dc
import numpy as np
from tensorflow.python.framework import test_util
from deepchem.models.tf_new_models.vina_model import VinaModel
from deepchem.models.tf_new_models.vina_model import get_cells
from deepchem.models.tf_new_models.vina_model import put_atoms_in_cells
from deepchem.models.tf_new_models.vina_model import compute_neighbor_cells
from deepchem.models.tf_new_models.vina_model import compute_closest_neighbors
from deepchem.models.tf_new_models.vina_model import get_cells_for_atoms
from deepchem.models.tf_new_models.vina_model import compute_neighbor_list
import deepchem.utils.rdkit_util as rdkit_util
from deepchem.utils.save import load_sdf_files
from deepchem.utils import pad_array


class TestVinaModel(test_util.TensorFlowTestCase):
  """
  Test Container usage.
  """

  def setUp(self):
    super(TestVinaModel, self).setUp()
    self.root = '/tmp'

  def test_vina_model(self):
    """Simple test that a vina model can be initialized."""
    vina_model = VinaModel()

  def test_get_cells(self):
    """Test that tensorflow can compute grid cells."""
    N = 10
    start = 0
    stop = 4
    nbr_cutoff = 1
    with self.test_session() as sess:
      ndim = 3
      cells = get_cells(start, stop, nbr_cutoff, ndim=ndim).eval()
      assert len(cells.shape) == 2
      assert cells.shape[0] == 4**ndim

      ndim = 2
      cells = get_cells(start, stop, nbr_cutoff, ndim=ndim).eval()
      assert len(cells.shape) == 2
      assert cells.shape[0] == 4**ndim

      # TODO(rbharath): Check that this operation is differentiable.

  def test_compute_neighbor_list(self):
    """Test that neighbor list can be computed with tensorflow"""
    N = 10
    start = 0
    stop = 12
    nbr_cutoff = 3
    ndim = 3
    M = 6
    k = 5
    # The number of cells which we should theoretically have
    n_cells = int(((stop - start) / nbr_cutoff)**ndim)

    with self.test_session() as sess:
      coords = start + np.random.rand(N, ndim) * (stop - start)
      coords = tf.stack(coords)
      nbr_list = compute_neighbor_list(
          coords, nbr_cutoff, N, M, n_cells, ndim=ndim, k=k)
      nbr_list = nbr_list.eval()
      assert nbr_list.shape == (N, M)

  def test_put_atoms_in_cells(self):
    """Test that atoms can be partitioned into spatial cells."""
    N = 10
    start = 0
    stop = 4
    nbr_cutoff = 1
    ndim = 3
    k = 5
    # The number of cells which we should theoretically have
    n_cells = ((stop - start) / nbr_cutoff)**ndim

    with self.test_session() as sess:
      cells = get_cells(start, stop, nbr_cutoff, ndim=ndim)
      coords = np.random.rand(N, ndim)
      _, atoms_in_cells = put_atoms_in_cells(coords, cells, N, n_cells, ndim, k)
      atoms_in_cells = atoms_in_cells.eval()
      assert len(atoms_in_cells) == n_cells
      # Each atom neighbors tensor should be (k, ndim) shaped.
      for atoms in atoms_in_cells:
        assert atoms.shape == (k, ndim)

  def test_compute_neighbor_cells(self):
    """Test that indices of neighboring cells can be computed."""
    N = 10
    start = 0
    stop = 4
    nbr_cutoff = 1
    ndim = 3
    # The number of cells which we should theoretically have
    n_cells = ((stop - start) / nbr_cutoff)**ndim

    # TODO(rbharath): The test below only checks that shapes work out.
    # Need to do a correctness implementation vs. a simple CPU impl.

    with self.test_session() as sess:
      cells = get_cells(start, stop, nbr_cutoff, ndim=ndim)
      nbr_cells = compute_neighbor_cells(cells, ndim, n_cells)
      nbr_cells = nbr_cells.eval()
      assert len(nbr_cells) == n_cells
      nbr_cells = [nbr_cell for nbr_cell in nbr_cells]
      for nbr_cell in nbr_cells:
        assert nbr_cell.shape == (26,)

  def test_compute_closest_neighbors(self):
    """Test that closest neighbors can be computed properly"""
    N = 10
    start = 0
    stop = 4
    nbr_cutoff = 1
    ndim = 3
    k = 5
    # The number of cells which we should theoretically have
    n_cells = ((stop - start) / nbr_cutoff)**ndim

    # TODO(rbharath): The test below only checks that shapes work out.
    # Need to do a correctness implementation vs. a simple CPU impl.

    with self.test_session() as sess:
      cells = get_cells(start, stop, nbr_cutoff, ndim=ndim)
      nbr_cells = compute_neighbor_cells(cells, ndim, n_cells)
      coords = np.random.rand(N, ndim)
      _, atoms_in_cells = put_atoms_in_cells(coords, cells, N, n_cells, ndim, k)
      nbrs = compute_closest_neighbors(coords, cells, atoms_in_cells, nbr_cells,
                                       N, n_cells)

  def test_get_cells_for_atoms(self):
    """Test that atoms are placed in the correct cells."""
    N = 10
    start = 0
    stop = 4
    nbr_cutoff = 1
    ndim = 3
    k = 5
    # The number of cells which we should theoretically have
    n_cells = ((stop - start) / nbr_cutoff)**ndim

    # TODO(rbharath): The test below only checks that shapes work out.
    # Need to do a correctness implementation vs. a simple CPU impl.

    with self.test_session() as sess:
      cells = get_cells(start, stop, nbr_cutoff, ndim=ndim)
      coords = np.random.rand(N, ndim)
      cells_for_atoms = get_cells_for_atoms(coords, cells, N, n_cells, ndim)
      cells_for_atoms = cells_for_atoms.eval()
      assert cells_for_atoms.shape == (N, 1)

  def test_vina_construct_graph(self):
    """Test that vina model graph can be constructed."""
    data_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(data_dir, "1jld_protein.pdb")
    ligand_file = os.path.join(data_dir, "1jld_ligand.pdb")

    vina_model = VinaModel()

  # TODO(rbharath): Commenting this out due to weird segfaults
  #def test_vina_generate_conformers(self):
  #  """Test that Vina Model can generate conformers"""
  #  data_dir = os.path.dirname(os.path.realpath(__file__))
  #  protein_file = os.path.join(data_dir, "1jld_protein.pdb")
  #  ligand_file = os.path.join(data_dir, "1jld_ligand.pdb")

  #  max_protein_atoms = 3500 
  #  max_ligand_atoms = 100

  #  print("Loading protein file")
  #  protein_xyz, protein_mol = rdkit_util.load_molecule(protein_file)
  #  protein_Z = pad_array(
  #      np.array([atom.GetAtomicNum() for atom in protein_mol.GetAtoms()]),
  #      max_protein_atoms)
  #  print("Loading ligand file")
  #  ligand_xyz, ligand_mol = rdkit_util.load_molecule(ligand_file)
  #  ligand_Z = pad_array(
  #      np.array([atom.GetAtomicNum() for atom in ligand_mol.GetAtoms()]),
  #      max_ligand_atoms)
