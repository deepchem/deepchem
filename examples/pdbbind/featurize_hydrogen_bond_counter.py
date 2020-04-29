# This example featurizes a small subset of the PDBBind v2015 protein-ligand core data with the AtomicConvFeaturizer
import deepchem as dc
import logging
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_files

complex_files = get_pdbbind_molecular_complex_files(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
n_featurize = 10
core_subset = complex_files[:n_featurize]

featurizer = dc.feat.HydrogenBondCounter()
features, failures = featurizer.featurize_complexes(
    core_subset, parallelize=False)
print("features.shape")
print(features.shape)
print("%d failures" % len(failures))
assert features.shape == (n_featurize, 3)
