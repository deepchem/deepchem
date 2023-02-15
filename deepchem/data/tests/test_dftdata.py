import pytest
try:
    from deepchem.data.dft_data import DFTEntry
except ModuleNotFoundError:
    raise ModuleNotFoundError("This utility requires dqc")
import torch
import yaml
from yaml.loader import SafeLoader


@pytest.mark.dqc
def test_entryDM():
    entry_path = 'dft_dset1.yaml'
    with open(entry_path) as f:
        data_mol = yaml.load(f, Loader=SafeLoader)
    dm_entry_for_HF = DFTEntry.create(data_mol[1])

    assert dm_entry_for_HF.entry_type == 'dm'
    dm_HF_system0 = dm_entry_for_HF.get_systems()[0]
    mol_dqc = dm_HF_system0.get_dqc_mol(dm_entry_for_HF)
    hf_zs = torch.Tensor([1, 9])
    hf_pos = torch.DoubleTensor([[0.86625, 0.0000, 0.0000],
                                 [-0.86625, 0.0000, 0.0000]])
    assert (mol_dqc.atomzs == hf_zs).all()
    assert (hf_pos.numpy() == mol_dqc.atompos.numpy()).all()
    dm0 = dm_entry_for_HF.get_true_val()
    assert dm0.shape == (57, 57)


@pytest.mark.dqc
def test_entryAE():
    entry_path = 'dft_dset1.yaml'
    with open(entry_path) as f:
        data_mol = yaml.load(f, Loader=SafeLoader)
    ae_entry_for_LiH = DFTEntry.create(data_mol[0])
    assert ae_entry_for_LiH.entry_type == 'ae'
    assert ae_entry_for_LiH.get_true_val() == 0.09194410469
