import pytest
import numpy as np
from deepchem.data import NumpyDataset

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_atomic_convolution_module():
    from deepchem.models.torch_models.layers import AtomicConv
    f1_num_atoms = 100  # maximum number of atoms to consider in the ligand
    f2_num_atoms = 1000  # maximum number of atoms to consider in the protein
    max_num_neighbors = 12  # maximum number of spatial neighbors for an atom

    acm = AtomicConv(
        n_tasks=1,
        frag1_num_atoms=f1_num_atoms,
        frag2_num_atoms=f2_num_atoms,
        complex_num_atoms=f1_num_atoms + f2_num_atoms,
        max_num_neighbors=max_num_neighbors,
        batch_size=12,
        layer_sizes=[32, 32, 16],
    )

    frag1_size = (acm._frag1_conv.size()[1]) * (acm._frag1_conv.size()[2])
    frag2_size = (acm._frag2_conv.size()[1]) * (acm._frag2_conv.size()[2])
    complex_size = (acm._complex_conv.size()[1]) * (acm._complex_conv.size()[2])

    assert acm.prev_layer.size() == torch.Size(
        [acm.batch_size, frag1_size + frag2_size + complex_size])


@pytest.mark.slow
@pytest.mark.torch
def test_atomic_conv_initialize_params():
    """Quick test of AtomConv."""
    from deepchem.models.torch_models import AtomConvModel
    acm = AtomConvModel(n_tasks=1,
                        batch_size=1,
                        layer_sizes=[
                            1,
                        ],
                        frag1_num_atoms=5,
                        frag2_num_atoms=5,
                        complex_num_atoms=10)

    assert acm.complex_num_atoms == 10
    assert len(acm.atom_types) == 15


@pytest.mark.slow
@pytest.mark.torch
def test_atomic_convolution_model():
    from deepchem.models.torch_models import AtomConvModel

    # For simplicity, let's assume both molecules have same number of
    # atoms.
    N_atoms = 5
    batch_size = 1
    atomic_convnet = AtomConvModel(n_tasks=1,
                                   batch_size=batch_size,
                                   layer_sizes=[10],
                                   frag1_num_atoms=N_atoms,
                                   frag2_num_atoms=N_atoms,
                                   complex_num_atoms=N_atoms * 2,
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
    atomic_convnet.fit(train, nb_epoch=200)
    preds = atomic_convnet.predict(train)
    assert np.allclose(labels, preds, atol=0.01)


@pytest.mark.slow
@pytest.mark.torch
def test_atomic_convolution_model_variable():
    """A simple test that initializes and fits an AtomConvModel on variable input size."""
    from deepchem.models.torch_models import AtomConvModel
    frag1_num_atoms = 100  # atoms for ligand
    frag2_num_atoms = 1200  # atoms for protein
    complex_num_atoms = frag1_num_atoms + frag2_num_atoms
    batch_size = 1
    atomic_convnet = AtomConvModel(n_tasks=1,
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
    assert preds.shape == (1, 1)
    assert np.count_nonzero(preds) > 0
