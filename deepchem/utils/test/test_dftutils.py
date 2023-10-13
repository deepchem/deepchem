"""
Test for DFT Utilities
"""
try:
    import dqc
    from dqc.system.mol import Mol
    from dqc.qccalc.ks import KS
    from deepchem.utils.dftutils import KSCalc, hashstr, SpinParam
    import torch
except ModuleNotFoundError:
    pass
import pytest


@pytest.mark.dqc
def test_dftutils():
    system = {
        'type': 'mol',
        'kwargs': {
            'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
            'basis': '6-311++G(3df,3pd)'
        }
    }
    atomzs, atomposs = dqc.parse_moldesc(system["kwargs"]["moldesc"])
    mol = Mol(**system["kwargs"])
    qc = KS(mol, xc='lda_x').run()
    qcs = KSCalc(qc)
    a = qcs.energy()
    b = torch.tensor(-99.1360, dtype=torch.float64)
    assert torch.allclose(a, b)


@pytest.mark.dqc
def test_SpinParam_sum():
    dens_u = torch.rand(10)
    dens_d = torch.rand(10)
    sp = SpinParam(u=dens_u, d=dens_d)

    assert torch.all(sp.sum().eq(dens_u + dens_d)).item()


@pytest.mark.dqc
def test_SpinParam_reduce():
    dens_u = torch.rand(10)
    dens_d = torch.rand(10)
    sp = SpinParam(u=dens_u, d=dens_d)

    def fcn(a, b):
        return a * b

    assert torch.all(sp.reduce(fcn).eq(dens_u * dens_d)).item()


@pytest.mark.dqc
def test_str():
    s = "hydrogen fluoride"
    s = hashstr(s)
    s1 = "df4e3775493a2e784618edaf9e96b7ecb6ce2b4cd022e8619588d55009872bb2"
    assert s == s1


@pytest.mark.dqc
def test_val_grad():
    """Test ValGrad data structure"""
    from deepchem.utils.dft_utils.datastruct import ValGrad

    vg1 = ValGrad(torch.tensor([1, 2, 3]))
    assert torch.allclose(vg1.value, torch.tensor([1, 2, 3]))
    vg1.grad = torch.tensor([4, 5, 6])
    assert torch.allclose(vg1.grad, torch.tensor([4, 5, 6]))
    vg2 = ValGrad(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    vg3 = vg1 + vg2
    assert torch.allclose(vg3.value, torch.tensor([2, 4, 6]))
    assert torch.allclose(vg3.grad, torch.tensor([8, 10, 12]))


@pytest.mark.dqc
def test_spin_param():
    """Test SpinParam data structure"""
    from deepchem.utils.dft_utils.datastruct import SpinParam

    sp = SpinParam(1, 2)
    assert sp.sum() == 3
    assert sp.reduce(lambda x, y: x * y + 2) == 4
