"""
Tests for Atomic Convolutions.
"""
from __future__ import division
from __future__ import unicode_literals

from nose.plugins.attrib import attr

import deepchem
import numpy as np
import tensorflow as tf
import unittest
import numpy as np
from deepchem.models.tensorgraph.models import atomic_conv
from deepchem.models.tensorgraph import layers
from deepchem.data import NumpyDataset


class TestAtomicConv(unittest.TestCase):

  @attr("slow")
  def test_atomic_conv(self):
    """A simple test that initializes and fits an AtomicConvModel."""
    # For simplicity, let's assume both molecules have same number of
    # atoms.
    N_atoms = 5
    batch_size = 1
    atomic_convnet = atomic_conv.AtomicConvModel(
        batch_size=batch_size,
        frag1_num_atoms=5,
        frag2_num_atoms=5,
        complex_num_atoms=10)

    # Creates a set of dummy features that contain the coordinate and
    # neighbor-list features required by the AtomicConvModel.
    features = []
    frag1_coords = np.random.rand(N_atoms, 3)
    frag1_nbr_list = {0: [], 1: [], 2: [], 3: [], 4: []}
    frag1_z = np.random.randint(10, size=(N_atoms))
    frag2_coords = np.random.rand(N_atoms, 3)
    frag2_nbr_list = {0: [], 1: [], 2: [], 3: [], 4: []}
    #frag2_z = np.random.rand(N_atoms, 3)
    frag2_z = np.random.randint(10, size=(N_atoms))
    system_coords = np.random.rand(2 * N_atoms, 3)
    system_nbr_list = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []
    }
    system_z = np.random.randint(10, size=(2 * N_atoms))

    features.append(
        (frag1_coords, frag1_nbr_list, frag1_z, frag2_coords, frag2_nbr_list,
         frag2_z, system_coords, system_nbr_list, system_z))
    features = np.asarray(features)
    labels = np.zeros(batch_size)
    train = NumpyDataset(features, labels)
    atomic_convnet.fit(train, nb_epoch=1)
