"""
Compare DeepChem LDA with LibXC LDA for H2 molecule.
"""
from deepchem.utils.dft_utils import Mol, KS
from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA, PyTorchGGA, PyTorchMGGA


def compare_lda_h2():
    moldesc = "H 0 0 -0.74; H 0 0 0.74"
    basis = "sto-3g"
    system = Mol(moldesc=moldesc, basis=basis)

    # DeepChem LDA_X
    print("\n1. DeepChem LDA (exchange only):")
    xc_dc = PyTorchLDA()
    ks_dc = KS(system, xc_dc, variational=False)
    ks_dc.run()
    e_dc = ks_dc.energy().item()
    print(f"   Total Energy: {e_dc:.6f} Ha")

    # LibXC LDA_X
    print("\n2. LibXC LDA (exchange only):")
    xc_libxc = "lda_x"
    ks_libxc = KS(system, xc_libxc, variational=False)
    ks_libxc.run()
    e_libxc = ks_libxc.energy().item()
    print(f"   Total Energy: {e_libxc:.6f} Ha")

    print(f"\n   Difference: {abs(e_dc - e_libxc):.6f} Ha")


def compare_gga_h2():
    moldesc = "H 0 0 -0.74; H 0 0 0.74"
    basis = "sto-3g"
    system = Mol(moldesc=moldesc, basis=basis)

    # DeepChem LDA_X
    print("\n1. DeepChem GGA (exchange only):")
    xc_dc = PyTorchGGA("gga_x_pbe")
    ks_dc = KS(system, xc_dc, variational=False)
    ks_dc.run()
    #e_dc = ks_dc.energy().item()
    e_dc = ks_dc.energy()
    #print(f"   Total Energy: {e_dc:.6f} Ha")
    print(f"   Total Energy: {e_dc} Ha")

    # LibXC LDA_X
    print("\n2. LibXC GGA (exchange only):")
    xc_libxc = "gga_x_pbe"
    ks_libxc = KS(system, xc_libxc, variational=False)
    ks_libxc.run()
    #e_libxc = ks_libxc.energy().item()
    e_libxc = ks_libxc.energy()
    #print(f"   Total Energy: {e_libxc:.6f} Ha")
    print(f"   Total Energy: {e_libxc} Ha")

    #print(f"\n   Difference: {abs(e_dc - e_libxc):.6f} Ha")
    print(f"\n   Difference: {abs(e_dc - e_libxc)} Ha")


def compare_mgga_h2():
    moldesc = "H 0 0 -0.74; H 0 0 0.74"
    basis = "sto-3g"
    system = Mol(moldesc=moldesc, basis=basis)

    # DeepChem LDA_X
    print("\n1. DeepChem MGGA (exchange only):")
    xc_dc = PyTorchMGGA()
    ks_dc = KS(system, xc_dc, variational=False)
    ks_dc.run()
    e_dc = ks_dc.energy().item()
    print(f"   Total Energy: {e_dc:.6f} Ha")

    # LibXC LDA_X
    print("\n2. LibXC MGGA (exchange only):")
    xc_libxc = "mgga_x_tpss"
    ks_libxc = KS(system, xc_libxc, variational=False)
    ks_libxc.run()
    e_libxc = ks_libxc.energy().item()
    print(f"   Total Energy: {e_libxc:.6f} Ha")

    print(f"\n   Difference: {abs(e_dc - e_libxc):.6f} Ha")

if __name__ == "__main__":
    compare_mgga_h2()
