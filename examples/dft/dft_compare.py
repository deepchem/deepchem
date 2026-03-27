"""
Compare DeepChem LDA with LibXC LDA for various molecules.
"""
from deepchem.utils.dft_utils import Mol, KS
from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA


def compare_lda(name, moldesc, basis="sto-3g"):
    """Compare DeepChem LDA vs LibXC LDA for a given molecule."""
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")

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
    return e_dc, e_libxc


def compare_lda_h2():
    moldesc = "H 0 0 -0.74; H 0 0 0.74"
    return compare_lda("H2 (Hydrogen)", moldesc)


def compare_lda_ch4():
    # CH4 tetrahedral geometry (bond length ~1.09 Angstrom)
    moldesc = (
        "C 0 0 0; "
        "H 0.6276 0.6276 0.6276; "
        "H -0.6276 -0.6276 0.6276; "
        "H -0.6276 0.6276 -0.6276; "
        "H 0.6276 -0.6276 -0.6276"
    )
    return compare_lda("CH4 (Methane)", moldesc)


def compare_lda_h2o():
    # H2O geometry (bond length ~0.96 Angstrom, angle ~104.5 degrees)
    moldesc = (
        "O 0 0 0; "
        "H 0.757 0.587 0; "
        "H -0.757 0.587 0"
    )
    return compare_lda("H2O (Water)", moldesc)


def compare_lda_nh3():
    # NH3 geometry (bond length ~1.01 Angstrom)
    moldesc = (
        "N 0 0 0.1173; "
        "H 0 0.9377 -0.2738; "
        "H 0.8121 -0.4689 -0.2738; "
        "H -0.8121 -0.4689 -0.2738"
    )
    return compare_lda("NH3 (Ammonia)", moldesc)


def compare_lda_lih():
    # LiH geometry (bond length ~1.60 Angstrom)
    moldesc = "Li 0 0 0; H 0 0 1.596"
    return compare_lda("LiH (Lithium Hydride)", moldesc)


if __name__ == "__main__":
    compare_lda_h2()
    compare_lda_ch4()
    compare_lda_h2o()
    compare_lda_nh3()
    compare_lda_lih()
