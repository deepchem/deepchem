import deepchem as dc
import numpy as np
import pytest

try:
    import torch
    import deepchem.models.torch_models.layers as torch_layers
    has_torch = True
except:
    has_torch = False


@pytest.mark.torch
def test_weave_gather_without_compression():
    """Test invoking the torch equivalent of WeaveGather."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    atom_feat = []
    atom_split = []
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        atom_split.extend([im] * n_atoms)

        # atom features
        atom_feat.append(mol.get_atom_features())
    inputs = [
        np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
        np.array(atom_split)
    ]
    torch.set_printoptions(precision=8)
    # Try without compression
    gather = torch_layers.WeaveGather(batch_size=2,
                                      n_input=75,
                                      gaussian_expand=True)
    # Outputs should be [mol1_vec, mol2_vec]
    outputs = gather(inputs)
    assert len(outputs) == 2
    assert np.array(outputs[0]).shape == (11 * 75,)
    assert np.array(outputs[1]).shape == (11 * 75,)
    assert np.allclose(
        outputs.numpy(),
        np.load(
            "deepchem/models/tests/assets/weavegather_results_without_compression.npy"
        ),
        atol=1e-4)


@pytest.mark.torch
def test_weave_gather_with_compression():
    """Test invoking the torch equivalent of WeaveGather."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    atom_feat = []
    atom_split = []
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        atom_split.extend([im] * n_atoms)

        # atom features
        atom_feat.append(mol.get_atom_features())
    inputs = [
        np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
        np.array(atom_split)
    ]
    torch.set_printoptions(precision=8)
    # Try with compression
    gather = torch_layers.WeaveGather(batch_size=2,
                                      n_input=75,
                                      gaussian_expand=True,
                                      compress_post_gaussian_expansion=True)
    gather.W = torch.from_numpy(
        np.load("deepchem/models/tests/assets/weavegather_weights.npy"))
    # Outputs should be [mol1_vec, mol2_vec]
    outputs = gather(inputs)
    assert len(outputs) == 2
    assert np.array(outputs[0]).shape == (75,)
    assert np.array(outputs[1]).shape == (75,)
    assert np.allclose(
        outputs.numpy(),
        np.load(
            "deepchem/models/tests/assets/weavegather_results_with_compression.npy"
        ),
        atol=1e-4)
