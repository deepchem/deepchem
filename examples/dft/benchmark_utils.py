"""Utilities for DFT bond-stretch benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto

AtomCoord = Tuple[str, np.ndarray]
Geometry = List[AtomCoord]


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    distances: np.ndarray
    reference_energies: np.ndarray
    deepchem_energies: np.ndarray
    reference_relative: np.ndarray
    deepchem_relative: np.ndarray
    absolute_error: np.ndarray
    mae: float
    max_error: float


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return a normalized version of the input vector."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vec / norm


def geometry_to_moldesc(geometry: Geometry) -> str:
    """Convert geometry into a semicolon-separated molecular description."""
    parts = []
    for atom, coord in geometry:
        x, y, z = coord
        parts.append(f"{atom} {x:.8f} {y:.8f} {z:.8f}")
    return "; ".join(parts)


def generate_h2o_oh_stretch_geometries(
    distances: Sequence[float],
    oxygen: Optional[np.ndarray] = None,
    hydrogen_moving: Optional[np.ndarray] = None,
    hydrogen_fixed: Optional[np.ndarray] = None,
) -> List[Geometry]:
    """Generate H2O geometries by stretching one O-H bond."""
    if oxygen is None:
        oxygen = np.array([0.0, 0.0, 0.0], dtype=float)
    if hydrogen_moving is None:
        hydrogen_moving = np.array([0.0, 0.7570, 0.5860], dtype=float)
    if hydrogen_fixed is None:
        hydrogen_fixed = np.array([0.0, -0.7570, 0.5860], dtype=float)

    unit_vec = normalize_vector(hydrogen_moving - oxygen)

    geometries: List[Geometry] = []
    for r in distances:
        moved_h = oxygen + float(r) * unit_vec
        geometry: Geometry = [
            ("O", oxygen.copy()),
            ("H", moved_h),
            ("H", hydrogen_fixed.copy()),
        ]
        geometries.append(geometry)

    return geometries


def compute_pyscf_ukS_energy(
    moldesc: str,
    basis: str = "6-31g",
    xc: str = "lda,vwn",
    charge: int = 0,
    spin: int = 0,
) -> float:
    """Compute total electronic energy using PySCF UKS."""
    mol = gto.M(
        atom=moldesc,
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-6
    mf.xc = xc
    energy = mf.kernel()
    return float(energy)


def shift_to_relative_energies(energies: np.ndarray) -> np.ndarray:
    """Shift energies so that the minimum is zero."""
    return energies - np.min(energies)


def evaluate_benchmark(
    distances: np.ndarray,
    geometries: List[Geometry],
    basis: str,
    pyscf_xc: str,
    deepchem_energy_fn: Callable[[str, str], float],
) -> BenchmarkResult:
    """Evaluate reference and DeepChem energies across all scan points."""
    reference_energies = []
    deepchem_energies = []

    for geometry in geometries:
        moldesc = geometry_to_moldesc(geometry)

        e_ref = compute_pyscf_ukS_energy(
            moldesc=moldesc,
            basis=basis,
            xc=pyscf_xc,
            charge=0,
            spin=0,
        )
        reference_energies.append(e_ref)

        e_dc = deepchem_energy_fn(moldesc, basis)
        deepchem_energies.append(e_dc)

    reference_energies_arr = np.asarray(reference_energies, dtype=float)
    deepchem_energies_arr = np.asarray(deepchem_energies, dtype=float)

    reference_relative = shift_to_relative_energies(reference_energies_arr)
    deepchem_relative = shift_to_relative_energies(deepchem_energies_arr)

    absolute_error = np.abs(reference_relative - deepchem_relative)

    return BenchmarkResult(
        distances=np.asarray(distances, dtype=float),
        reference_energies=reference_energies_arr,
        deepchem_energies=deepchem_energies_arr,
        reference_relative=reference_relative,
        deepchem_relative=deepchem_relative,
        absolute_error=absolute_error,
        mae=float(np.mean(absolute_error)),
        max_error=float(np.max(absolute_error)),
    )


def save_benchmark_csv(result: BenchmarkResult, output_csv: str) -> None:
    """Save benchmark results to a CSV file."""
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "distance_angstrom",
            "reference_energy_hartree",
            "deepchem_energy_hartree",
            "reference_relative_hartree",
            "deepchem_relative_hartree",
            "absolute_error_hartree",
        ])
        for i in range(len(result.distances)):
            writer.writerow([
                result.distances[i],
                result.reference_energies[i],
                result.deepchem_energies[i],
                result.reference_relative[i],
                result.deepchem_relative[i],
                result.absolute_error[i],
            ])


def plot_benchmark_curve(
    result: BenchmarkResult,
    output_png: Optional[str] = None,
    title: str = "H2O O-H Bond Stretch Benchmark",
) -> None:
    """Plot the benchmark curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(result.distances, result.reference_relative, marker="o", label="PySCF")
    plt.plot(result.distances, result.deepchem_relative, marker="s", label="DeepChem")
    plt.xlabel("O-H bond length (Angstrom)")
    plt.ylabel("Relative energy (Hartree)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_png is not None:
        output_dir = os.path.dirname(output_png)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_png, dpi=300)

    plt.show()


def print_benchmark_summary(result: BenchmarkResult) -> None:
    """Print a short benchmark summary."""
    ref_min_idx = int(np.argmin(result.reference_energies))
    dc_min_idx = int(np.argmin(result.deepchem_energies))

    print("\nBenchmark Summary")
    print("-" * 60)
    print(f"Reference minimum distance : {result.distances[ref_min_idx]:.4f} Angstrom")
    print(f"DeepChem minimum distance  : {result.distances[dc_min_idx]:.4f} Angstrom")
    print(f"MAE                        : {result.mae:.8f} Hartree")
    print(f"Max error                  : {result.max_error:.8f} Hartree")
    print("-" * 60)