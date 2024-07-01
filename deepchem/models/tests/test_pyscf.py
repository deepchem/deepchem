import pytest


@pytest.mark.torch
def test_pyscf():
    import pyscf
    mol_h2o = pyscf.gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1',
                          parse_arg=False,
                          basis='ccpvdz')
    assert mol_h2o.basis == 'ccpvdz'
