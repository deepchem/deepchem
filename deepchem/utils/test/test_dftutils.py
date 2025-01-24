"""
Test for DFT Utilities
"""
import pytest
import warnings
try:
    import torch
except Exception as e:
    warnings.warn("Could not import torch. Skipping tests." + str(e))


@pytest.mark.torch
def test_dftutils():
    from deepchem.utils.dft_utils import parse_moldesc, Mol, KS
    from deepchem.utils.dftutils import KSCalc
    system = {
        'type': 'mol',
        'kwargs': {
            'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
            'basis': '6-311++G(3df,3pd)'
        }
    }
    atomzs, atomposs = parse_moldesc(system["kwargs"]["moldesc"])
    mol = Mol(**system["kwargs"], spin=0.0)
    qc = KS(mol, xc='lda_x').run()
    qcs = KSCalc(qc)
    a = qcs.energy()
    b = torch.tensor(-99.1360, dtype=torch.float64)
    assert torch.allclose(a, b)


@pytest.mark.torch
def test_SpinParam_sum():
    from deepchem.utils.dftutils import SpinParam
    dens_u = torch.rand(10)
    dens_d = torch.rand(10)
    sp = SpinParam(u=dens_u, d=dens_d)

    assert torch.all(sp.sum().eq(dens_u + dens_d)).item()


@pytest.mark.torch
def test_SpinParam_reduce():
    from deepchem.utils.dftutils import SpinParam
    dens_u = torch.rand(10)
    dens_d = torch.rand(10)
    sp = SpinParam(u=dens_u, d=dens_d)

    def fcn(a, b):
        return a * b

    assert torch.all(sp.reduce(fcn).eq(dens_u * dens_d)).item()


@pytest.mark.torch
def test_str():
    from deepchem.utils.dftutils import hashstr
    s = "hydrogen fluoride"
    s = hashstr(s)
    s1 = "df4e3775493a2e784618edaf9e96b7ecb6ce2b4cd022e8619588d55009872bb2"
    assert s == s1
