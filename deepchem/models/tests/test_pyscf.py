import pytest

@pytest.mark.dqc
def test_pyscf():
    import pyscf
    mol_h2o = pyscf.gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz')
    assert mol_h2o.basis == 'ccpvdz'
