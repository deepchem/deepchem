# This example featurizes a small subset of the PDBBind v2015 protein-ligand core data with the AtomicConvFeaturizer
import deepchem as dc
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_files

complex_files = get_pdbbind_molecular_complex_files(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
core_subset = complex_files[:2]
ligand_files = [core[1] for core in core_subset]
protein_files = [core[0] for core in core_subset]

frag1_num_atoms = 70  # for ligand atoms
frag2_num_atoms = 24000  # for protein atoms
complex_num_atoms = 24070  # in total
max_num_neighbors = 4
# Cutoff in angstroms
neighbor_cutoff = 4
featurizer = dc.feat.AtomicConvFeaturizer(
    frag_num_atoms=[frag1_num_atoms,frag2_num_atoms],
    complex_num_atoms=complex_num_atoms,
    max_num_neighbors=max_num_neighbors,
    neighbor_cutoff=neighbor_cutoff)
features, failures = featurizer.featurize_complexes(
    ligand_files, protein_files, parallelize=False)
