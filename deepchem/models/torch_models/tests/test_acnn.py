import pytest

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
