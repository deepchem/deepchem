import pytest
try:
    from deepchem.data.dft_data import DFTEntry
except ModuleNotFoundError:
    raise ModuleNotFoundError("This utility requires dqc")
import torch


@pytest.mark.dqc
def test_entryDM():
    data_mol = {
        'name':
            'Density matrix of HF',
        'type':
            'dm',
        'cmd':
            'dm(systems[0])',
        'true_val':
            'output.npy',
        'systems': [{
            'type': 'mol',
            'kwargs': {
                'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
                'basis': '6-311++G(3df,3pd)'
            }
        }]
    }
    dm_entry_for_HF = DFTEntry.create(data_mol)
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
    data_mol = {
        'name':
            'Atomization energy of LiH',
        'type':
            'ae',
        'cmd':
            'energy(systems[1]) + energy(systems[2]) - energy(systems[0])',
        'true_val':
            0.09194410469,
        'systems': [{
            'type': 'mol',
            'kwargs': {
                'moldesc': 'Li 1.5070 0 0; H -1.5070 0 0',
                'basis': '6-311++G(3df,3pd)'
            }
        }, {
            'type': 'mol',
            'kwargs': {
                'moldesc': 'Li 0 0 0',
                'basis': '6-311++G(3df,3pd)',
                'spin': 1
            }
        }, {
            'type': 'mol',
            'kwargs': {
                'moldesc': 'H 0 0 0',
                'basis': '6-311++G(3df,3pd)',
                'spin': 1
            }
        }]
    }
    ae_entry_for_LiH = DFTEntry.create(data_mol)
    assert ae_entry_for_LiH.entry_type == 'ae'
    assert ae_entry_for_LiH.get_true_val() == 0.09194410469
