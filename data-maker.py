import torch
from deepchem.feat.dft_data import DFTSystem
from deepchem.utils.dft_utils.qccalc.ks import KS


def runner_deepchem(atoms):
    alattice = torch.tensor(atoms.get_cell())
    
    moldesc = "; ".join(
        f"{a} {x * 1.5} {y} {z}"
        for (a, [x, y, z]) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )

    system = {'moldesc': moldesc, 'basis': 'sto-3g', 'alattice': alattice}
    sol = DFTSystem(system).get_dqc_mol()

    target = KS(sol, xc='gga_x_pbe+gga_c_pbe').run(fwd_options={"maxiter": 100}).energy().real
    print(f"Target: {target:.6f} Ha")
