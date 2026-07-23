"""Run H2O O-H bond stretch benchmark with DeepChem and PySCF."""

from __future__ import annotations

import os
import numpy as np
import torch



from benchmark_utils import (
    evaluate_benchmark,
    generate_h2o_oh_stretch_geometries,
    plot_benchmark_curve,
    print_benchmark_summary,
    save_benchmark_csv,
)

from deepchem.feat.dft_data import DFTEntry, DFTSystem
from deepchem.models.dft.nnxc import HybridXC
from deepchem.models.dft.scf import XCNNSCF

nnmodel = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1),
    ).to(torch.double)

for param in nnmodel.parameters():
        torch.nn.init.zeros_(param)


def build_dummy_entry(moldesc: str, basis: str) -> DFTEntry:
    """Create a minimal DFTEntry for SCF energy evaluation.

    This follows the same construction style shown in the notebook.
    """
    e_type = "dm"
    true_val = 0.0
    systems = [{"moldesc": moldesc, "basis": basis}]
    return DFTEntry.create(e_type, true_val, systems)


def compute_deepchem_energy(
    moldesc: str,
    basis: str = "6-31g",
) -> float:
    """Compute total energy using DeepChem XCNNSCF.

    This follows the notebook pattern:
    1. build systems
    2. build DFTEntry
    3. build DFTSystem
    4. build HybridXC
    5. run SCF
    6. return energy
    """
    systems = [{"moldesc": moldesc, "basis": basis}]
    entry = build_dummy_entry(moldesc=moldesc, basis=basis)
    system = DFTSystem(systems[0])



    hybrid_xc = HybridXC("lda_x + lda_c_vwn", nnmodel, aweight0=0.0)

    scf_evaluator = XCNNSCF(hybrid_xc, entry)
    run_result = scf_evaluator.run(system)
    energy = run_result.energy()

    if hasattr(energy, "detach"):
        energy = energy.detach().cpu().item()
    elif hasattr(energy, "item"):
        energy = energy.item()

    return float(energy)


def main() -> None:
    """Main benchmark runner."""
    distances = np.linspace(0.75, 3.00, 25)
    geometries = generate_h2o_oh_stretch_geometries(distances=distances)

    output_dir = os.path.join("outputs", "h2o_benchmark")
    output_csv = os.path.join(output_dir, "h2o_oh_stretch_results.csv")
    output_png = os.path.join(output_dir, "h2o_oh_stretch_curve.png")

    result = evaluate_benchmark(
        distances=distances,
        geometries=geometries,
        basis="6-31g",
        pyscf_xc="lda,vwn",
        deepchem_energy_fn=compute_deepchem_energy,
    )

    save_benchmark_csv(result, output_csv)
    plot_benchmark_curve(
        result=result,
        output_png=output_png,
        title="H2O O-H Bond Stretch: DeepChem vs PySCF",
    )
    print_benchmark_summary(result)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()