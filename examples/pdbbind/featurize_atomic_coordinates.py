# This example featurizes a small subset of the PDBBind v2015 protein-ligand core data with the AtomicConvFeaturizer
import deepchem as dc
import logging
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_files

complex_files = get_pdbbind_molecular_complex_files(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
ligand_files = [complex[1] for complex in complex_files]
core_subset = ligand_files[:2]

featurizer = dc.feat.AtomicCoordinates()
features = featurizer.featurize(core_subset)
assert features.shape == (2,)
