import os
import pathlib
import pytest
from deepchem.data.dft_data import DFTEntry, DFTSystem, load_entries
import torch 

@pytest.fixture
def entries():
    entries = load_entries('deepchem/data/tests/dft_dset1.yaml', device='cpu')
    return entries

@pytest.fixture
def dm_entry_for_HF(entries):
    # Density Matrix entry of HF
    entry = entries[1]
    return entry


@pytest.fixture
def dm_HF_system0(dm_entry_for_HF):
    system = dm_entry_for_HF.get_systems()[0]
    return system

@pytest.fixture
def ae_entry_for_LiH(entries):
    # AE entry of LiH
    entry = entries[0]
    return entry


@pytest.fixture
def ae_LiH_system0(ae_entry_for_LiH):
    system = ae_entry_for_LiH.get_systems()[0]
    return system

def test_entrytype(dm_entry_for_HF,ae_entry_for_LiH):
  assert dm_entry_for_HF.entry_type == 'dm'
  assert ae_entry_for_LiH.entry_type == 'ae'

def test_dqcsystem(dm_HF_system0, dm_entry_for_HF):
  mol_dqc = dm_HF_system0.get_dqc_system(dm_entry_for_HF)
  hf_zs = torch.Tensor([1, 9])
  hf_pos = torch.DoubleTensor([[0.86625, 0.0000, 0.0000],
                               [-0.86625, 0.0000, 0.0000]])
  assert (mol_dqc.atomzs == hf_zs).all()
  assert torch.allclose(hf_pos, mol_dqc.atompos)

def test_trueval(ae_entry_for_LiH):
  assert ae_entry_for_LiH.get_true_val() == 0.09194410469
