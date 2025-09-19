import time
from deepchem.utils.convert import smiles_to_iupac, iupac_to_smiles

test_data = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "C1=CC=CC=C1",  # benzene
    "C[C@@H](O)[C@H](O)CO",  # stereochem test
]

print("Benchmarking SMILES to IUPAC...")
for smiles in test_data:
    start = time.time()
    iupac = smiles_to_iupac(smiles)
    end = time.time()
    print(f"{smiles} -> {iupac} ({end - start:.4f}s)")

print("\nBenchmarking IUPAC to SMILES...")
for smiles in test_data:
    iupac = smiles_to_iupac(smiles)
    if iupac:
        start = time.time()
        recovered = iupac_to_smiles(iupac)
        end = time.time()
        print(f"{iupac} -> {recovered} ({end - start:.4f}s)")
