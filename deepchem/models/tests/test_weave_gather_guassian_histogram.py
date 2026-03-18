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
def test_weave_gather_gaussian_histogram():
    """Test invoking the torch equivalent of Gaussian Histograms."""
    from rdkit import Chem
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    gather = torch_layers.WeaveGather(batch_size=2, n_input=75)
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
    outputs = gather.gaussian_histogram(inputs[0])
    # Gaussian histograms expands into 11 Gaussian buckets.
    assert np.array(outputs).shape == (
        4,
        11 * 75,
    )
    assert np.allclose(
        outputs.numpy(),
        np.load("deepchem/models/tests/assets/gaussian_histogram_outputs.npy"),
        atol=1e-4)
