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
def test_weave_layer():
    """Test invoking the torch equivalent of WeaveLayer."""
    n_atoms = 4  # In CCC and C, there are 4 atoms
    raw_smiles = ['CCC', 'C']
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in raw_smiles]
    featurizer = dc.feat.WeaveFeaturizer()
    mols = featurizer.featurize(mols)
    weave = torch_layers.WeaveLayer()
    atom_feat = []
    pair_feat = []
    atom_to_pair = []
    pair_split = []
    start = 0
    n_pair_feat = 14
    for im, mol in enumerate(mols):
        n_atoms = mol.get_num_atoms()
        # index of pair features
        C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
        atom_to_pair.append(
            np.transpose(np.array([C1.flatten() + start,
                                   C0.flatten() + start])))
        # number of pairs for each atom
        pair_split.extend(C1.flatten() + start)
        start = start + n_atoms

        # atom features
        atom_feat.append(mol.get_atom_features())
        # pair features
        pair_feat.append(
            np.reshape(mol.get_pair_features(),
                       (n_atoms * n_atoms, n_pair_feat)))
    inputs = [
        np.array(np.concatenate(atom_feat, axis=0), dtype=np.float32),
        np.concatenate(pair_feat, axis=0),
        np.array(pair_split),
        np.concatenate(atom_to_pair, axis=0)
    ]
    torch.set_printoptions(precision=8)
    # Assigning tensorflow equivalent weights to torch layer
    weave.W_AA = torch.from_numpy(
        np.load("deepchem/models/tests/assets/W_AA.npy"))
    weave.W_PA = torch.from_numpy(
        np.load("deepchem/models/tests/assets/W_PA.npy"))
    weave.W_A = torch.from_numpy(
        np.load("deepchem/models/tests/assets/W_A.npy"))
    if weave.update_pair:
        weave.W_AP = torch.from_numpy(
            np.load("deepchem/models/tests/assets/W_AP.npy"))
        weave.W_PP = torch.from_numpy(
            np.load("deepchem/models/tests/assets/W_PP.npy"))
        weave.W_P = torch.from_numpy(
            np.load("deepchem/models/tests/assets/W_P.npy"))
    # Outputs should be [A, P]
    outputs = weave(inputs)
    assert len(outputs) == 2
    assert np.allclose(outputs[0].detach().numpy(),
                       np.load("deepchem/models/tests/assets/A.npy"),
                       atol=1e-4)
    assert np.allclose(outputs[1].detach().numpy(),
                       np.load("deepchem/models/tests/assets/P.npy"),
                       atol=1e-4)
