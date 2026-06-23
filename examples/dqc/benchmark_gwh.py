import torch
import time

# Direct routing to bypass deepchem's internal abstraction layers
from deepchem.utils.dft_utils.system.mol import Mol as mol
from deepchem.utils.dft_utils.qccalc.hf import HF

def run_scf_benchmark(dm0_guess):
    print(f"--- Running SCF with {dm0_guess.upper()} Guess ---")
    
    # 1. Define a Water Molecule (H2O)
    # Using standard coordinates and a minimal basis set (sto-3g) for rapid testing
    h2o_coords = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"
    system = mol(h2o_coords, basis="sto-3g", dtype=torch.float64)
    
    # 2. Initialize the Hartree-Fock calculation (no XC functional needed)
    qc = HF(system)
    
    # 3. Run the SCF optimization and time it
    start_time = time.time()
    
    # We pass your custom guess and turn on verbose logging to see the iteration count
    qc.run(
        dm0=dm0_guess, 
        fwd_options={"verbose": True}
    )
    
    end_time = time.time()
    
    # 4. Extract Results
    energy = qc.energy().item()
    print(f"Converged Energy: {energy:.6f} Hartree")
    print(f"Compute Time: {end_time - start_time:.4f} seconds\n")
    
    return energy

if __name__ == "__main__":
    print("Starting Initial Guess Optimization Benchmark...\n")
    
    # Run the baseline (the old way)
    energy_1e = run_scf_benchmark("1e")
    
    # Run your new architecture
    energy_gwh = run_scf_benchmark("gwh")
    
    # 5. Verify Quantum Mechanical Consistency
    diff = abs(energy_1e - energy_gwh)
    print(f"Energy Difference: {diff:.2e} Hartree")
    
    if diff < 1e-5:
        print("✅ SUCCESS: Both guesses converged to the same physical ground state!")
    else:
        print("❌ WARNING: Energies differ. Check the GWH tensor diagonal mappings.")