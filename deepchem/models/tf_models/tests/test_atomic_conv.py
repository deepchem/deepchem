"""
Tests for Atomic Convolutions.
"""

import os
import pytest
from flaky import flaky
import numpy as np
from deepchem.data import NumpyDataset
from deepchem.feat import AtomicConvFeaturizer

try:
    import tensorflow as tf  # noqa: F401
    from deepchem.models import atomic_conv
    has_tensorflow = True
except:
    has_tensorflow = False


@pytest.mark.tensorflow
def test_atomic_conv_initialize():
    """Quick test of AtomicConv."""
    acm = atomic_conv.AtomicConvModel(n_tasks=1,
                                      batch_size=1,
                                      layer_sizes=[
                                          1,
                                      ],
                                      frag1_num_atoms=5,
                                      frag2_num_atoms=5,
                                      complex_num_atoms=10)

    assert acm.complex_num_atoms == 10
    assert len(acm.atom_types) == 15


@flaky
@pytest.mark.slow
@pytest.mark.tensorflow
def test_atomic_conv():
    """A simple test that initializes and fits an AtomicConvModel."""
    # For simplicity, let's assume both molecules have same number of
    # atoms.
    N_atoms = 5
    batch_size = 1
    atomic_convnet = atomic_conv.AtomicConvModel(n_tasks=1,
                                                 batch_size=batch_size,
                                                 layer_sizes=[10],
                                                 frag1_num_atoms=5,
                                                 frag2_num_atoms=5,
                                                 complex_num_atoms=10,
                                                 dropouts=0.0,
                                                 learning_rate=0.003)

    # Creates a set of dummy features that contain the coordinate and
    # neighbor-list features required by the AtomicConvModel.
    features = []
    frag1_coords = np.random.rand(N_atoms, 3)
    frag1_nbr_list = {0: [], 1: [], 2: [], 3: [], 4: []}
    frag1_z = np.random.randint(10, size=(N_atoms))
    frag2_coords = np.random.rand(N_atoms, 3)
    frag2_nbr_list = {0: [], 1: [], 2: [], 3: [], 4: []}
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
    features = np.asarray(features, dtype=object)
    labels = np.random.rand(batch_size)
    train = NumpyDataset(features, labels)
    atomic_convnet.fit(train, nb_epoch=150)
    assert np.allclose(labels, atomic_convnet.predict(train), atol=0.01)


@pytest.mark.slow
@pytest.mark.tensorflow
def test_atomic_conv_variable():
    """A simple test that initializes and fits an AtomicConvModel on variable input size."""
    frag1_num_atoms = 1000
    frag2_num_atoms = 1200
    complex_num_atoms = frag1_num_atoms + frag2_num_atoms
    batch_size = 1
    atomic_convnet = atomic_conv.AtomicConvModel(
        n_tasks=1,
        batch_size=batch_size,
        layer_sizes=[
            10,
        ],
        frag1_num_atoms=frag1_num_atoms,
        frag2_num_atoms=frag2_num_atoms,
        complex_num_atoms=complex_num_atoms)

    # Creates a set of dummy features that contain the coordinate and
    # neighbor-list features required by the AtomicConvModel.
    features = []
    frag1_coords = np.random.rand(frag1_num_atoms, 3)
    frag1_nbr_list = {i: [] for i in range(frag1_num_atoms)}
    frag1_z = np.random.randint(10, size=(frag1_num_atoms))
    frag2_coords = np.random.rand(frag2_num_atoms, 3)
    frag2_nbr_list = {i: [] for i in range(frag2_num_atoms)}
    frag2_z = np.random.randint(10, size=(frag2_num_atoms))
    system_coords = np.random.rand(complex_num_atoms, 3)
    system_nbr_list = {i: [] for i in range(complex_num_atoms)}
    system_z = np.random.randint(10, size=(complex_num_atoms))

    features.append(
        (frag1_coords, frag1_nbr_list, frag1_z, frag2_coords, frag2_nbr_list,
         frag2_z, system_coords, system_nbr_list, system_z))
    features = np.asarray(features, dtype=object)
    labels = np.zeros(batch_size)
    train = NumpyDataset(features, labels)
    atomic_convnet.fit(train, nb_epoch=1)
    preds = atomic_convnet.predict(train)
    assert preds.shape == (1, 1, 1)
    assert np.count_nonzero(preds) > 0


@pytest.mark.slow
@pytest.mark.tensorflow
def test_atomic_conv_with_feat():
    """A simple test for running an atomic convolution on featurized data."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(dir_path,
                               "../../feat/tests/data/3zso_ligand_hyd.pdb")
    protein_file = os.path.join(dir_path,
                                "../../feat/tests/data/3zso_protein_noH.pdb")
    # Pulled from PDB files. For larger datasets with more PDBs, would use
    # max num atoms instead of exact.
    frag1_num_atoms = 44  # for ligand atoms
    frag2_num_atoms = 2334  # for protein atoms
    complex_num_atoms = 2378  # in total
    max_num_neighbors = 4
    # Cutoff in angstroms
    neighbor_cutoff = 4
    complex_featurizer = AtomicConvFeaturizer(frag1_num_atoms, frag2_num_atoms,
                                              complex_num_atoms,
                                              max_num_neighbors,
                                              neighbor_cutoff)
    # arbitrary label
    labels = np.array([0])
    features = complex_featurizer.featurize([(ligand_file, protein_file)])
    dataset = NumpyDataset(features, labels)

    batch_size = 1
    print("Constructing Atomic Conv model")
    atomic_convnet = atomic_conv.AtomicConvModel(
        n_tasks=1,
        batch_size=batch_size,
        layer_sizes=[10],
        frag1_num_atoms=frag1_num_atoms,
        frag2_num_atoms=frag2_num_atoms,
        complex_num_atoms=complex_num_atoms)

    print("About to call fit")
    # Run a fitting operation
    atomic_convnet.fit(dataset)
    preds = atomic_convnet.predict(dataset)
    assert preds.shape == (1, 1, 1)
    assert np.count_nonzero(preds) > 0
