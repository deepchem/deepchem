import os
import unittest
import tempfile

import scipy
import numpy as np

from deepchem.models import ANIRegression
import deepchem as dc


class TestANIRegression(unittest.TestCase):

  def setUp(self):

    max_atoms = 4

    X = np.array([[
        [1, 5.0, 3.2, 1.1],
        [6, 1.0, 3.4, -1.1],
        [1, 2.3, 3.4, 2.2],
        [0, 0, 0, 0],
    ], [
        [8, 2.0, -1.4, -1.1],
        [7, 6.3, 2.4, 3.2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]])

    y = np.array([2.0, 1.1])

    layer_structures = [128, 128, 64]
    atom_number_cases = [1, 6, 7, 8]

    self.model_dir = tempfile.mkdtemp()

    self.kwargs = {
        "n_tasks": 1,
        "max_atoms": max_atoms,
        "layer_structures": layer_structures,
        "atom_number_cases": atom_number_cases,
        "batch_size": 2,
        "learning_rate": 0.001,
        "use_queue": False,
        "mode": "regression",
        "model_dir": self.model_dir
    }

    model = ANIRegression(**self.kwargs)

    train_dataset = dc.data.NumpyDataset(X, y, n_tasks=1)

    model.fit(train_dataset, nb_epoch=2, checkpoint_interval=100)

    self.model = model

  def test_gradients(self):

    new_x = np.array([
        -2.0,
        1.2,
        2.1,
        1.3,
        -6.4,
        3.1,
        -2.5,
        2.4,
        5.6,
    ])

    new_atomic_nums = np.array([1, 1, 6])

    delta = 1e-2

    # use central difference since forward difference has a pretty high
    # approximation error

    grad_approx = []

    for idx in range(new_x.shape[0]):
      d_new_x_plus = np.array(new_x)
      d_new_x_plus[idx] += delta
      d_new_x_minus = np.array(new_x)
      d_new_x_minus[idx] -= delta
      dydx = (self.model.pred_one(d_new_x_plus, new_atomic_nums) -
              self.model.pred_one(d_new_x_minus, new_atomic_nums)) / (2 * delta)
      grad_approx.append(dydx[0])

    grad_approx = np.array(grad_approx)

    grad_exact = self.model.grad_one(new_x, new_atomic_nums)

    np.testing.assert_array_almost_equal(grad_approx, grad_exact, decimal=3)

    grad_exact_constrained = self.model.grad_one(
        new_x, new_atomic_nums, constraints=[0, 2])

    assert grad_exact_constrained[0] == 0
    assert grad_exact_constrained[1] == 0
    assert grad_exact_constrained[2] == 0

    assert grad_exact_constrained[3] == grad_exact[3]
    assert grad_exact_constrained[4] == grad_exact[4]
    assert grad_exact_constrained[5] == grad_exact[5]

    assert grad_exact_constrained[6] == 0
    assert grad_exact_constrained[7] == 0
    assert grad_exact_constrained[8] == 0

    min_coords = self.model.minimize_structure(
        new_x, new_atomic_nums, constraints=[0, 2])

    assert min_coords[0][0] == new_x[0]
    assert min_coords[0][1] == new_x[1]
    assert min_coords[0][2] == new_x[2]

    # assert min_coords[1][0] != new_x[3]
    # assert min_coords[1][1] != new_x[4]
    # assert min_coords[1][2] != new_x[5]

    assert min_coords[2][0] == new_x[6]
    assert min_coords[2][1] == new_x[7]
    assert min_coords[2][2] == new_x[8]

  def test_numpy_save_load(self):

    self.model.save_numpy()
    restored_model = ANIRegression.load_numpy(self.model_dir)

    new_x = np.array([
        -2.0,
        1.2,
        2.1,
        1.3,
        -6.4,
        3.1,
        -2.5,
        2.4,
        5.6,
    ])

    new_atomic_nums = np.array([1, 1, 6])

    expected = self.model.pred_one(new_x, new_atomic_nums)
    predicted = restored_model.pred_one(new_x, new_atomic_nums)

    assert self.model.n_tasks == restored_model.n_tasks
    assert self.model.max_atoms == restored_model.max_atoms
    assert self.model.layer_structures == restored_model.layer_structures
    assert self.model.atom_number_cases == restored_model.atom_number_cases
    assert self.model.batch_size == restored_model.batch_size
    assert self.model.use_queue == restored_model.use_queue

    assert expected == predicted


if __name__ == '__main__':
  unittest.main()
