# This example featurizes a small subset of the PDBBind v2015 protein-ligand core data with the AtomicConvFeaturizer
import deepchem as dc
import logging
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_files

complex_files = get_pdbbind_molecular_complex_files(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
n_complexes = len(complex_files)
n_featurize = 2
core_subset = complex_files[:n_featurize]

max_num_neighbors = 4
# Cutoff in angstroms
neighbor_cutoff = 4
featurizer = dc.feat.AtomicConvFeaturizer(
    frag_max_atoms=[100, 50],
    max_num_neighbors=max_num_neighbors,
    neighbor_cutoff=neighbor_cutoff)
features, failures = featurizer.featurize_complexes(
    core_subset, parallelize=False)

print("features.shape")
print(features.shape)
print("%d failures" % len(failures))
assert features.shape == (n_featurize -len(failures), 2)
