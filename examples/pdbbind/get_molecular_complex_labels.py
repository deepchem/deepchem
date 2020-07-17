# This example shows how to get the labels for PDBBind datasets.
import deepchem as dc
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_labels

pdbbind_v2015_core_labels = get_pdbbind_molecular_complex_labels(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
print("Number of labels in PDBBind v2015 core set: %d" % len(pdbbind_v2015_core_labels))

pdbbind_v2015_refined_labels = get_pdbbind_molecular_complex_labels(subset="refined", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
print("Number of labels in PDBBind v2015 refined set: %d" % len(pdbbind_v2015_refined_labels))

pdbbind_v2015_general_labels = get_pdbbind_molecular_complex_labels(subset="general", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
print("Number of labels in PDBBind v2015 general set: %d" % len(pdbbind_v2015_general_labels))

pdbbind_v2019_PP_labels = get_pdbbind_molecular_complex_labels(subset="general", version="v2019", interactions="protein-protein", load_binding_pocket=False)
print("Number of labels in PDBBind v2019 PP set: %d" % len(pdbbind_v2019_PP_labels))

pdbbind_v2019_PN_labels = get_pdbbind_molecular_complex_labels(subset="general", version="v2019", interactions="protein-nucleic-acid", load_binding_pocket=False)
print("Number of labels in PDBBind v2019 PN set: %d" % len(pdbbind_v2019_PN_labels))

pdbbind_v2019_NL_labels = get_pdbbind_molecular_complex_labels(subset="general", version="v2019", interactions="nucleic-acid-ligand", load_binding_pocket=False)
print("Number of labels in PDBBind v2019 NL set: %d" % len(pdbbind_v2019_NL_labels))

pdbbind_v2019_refined_labels = get_pdbbind_molecular_complex_labels(subset="refined", version="v2019", interactions="protein-ligand", load_binding_pocket=False)
print("Number of labels in PDBBind v2019 refined set: %d" % len(pdbbind_v2019_refined_labels))

pdbbind_v2019_general_labels = get_pdbbind_molecular_complex_labels(subset="general", version="v2019", interactions="protein-ligand", load_binding_pocket=False)
print("Number of labels in PDBBind v2019 general set: %d" % len(pdbbind_v2019_general_labels))
