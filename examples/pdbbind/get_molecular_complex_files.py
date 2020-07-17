# This examples shows how to get the list of structure files for PDBBind datasets
import deepchem as dc
from deepchem.molnet.load_function.pdbbind_datasets import get_pdbbind_molecular_complex_files

pdbbind_v2015_core_files = get_pdbbind_molecular_complex_files(subset="core", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
print("Number of files in PDBBind v2015 core set: %d" % len(pdbbind_v2015_core_files))

pdbbind_v2015_refined_files = get_pdbbind_molecular_complex_files(subset="refined", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
print("Number of files in PDBBind v2015 refined set: %d" % len(pdbbind_v2015_refined_files))

pdbbind_v2015_general_files = get_pdbbind_molecular_complex_files(subset="general", version="v2015", interactions="protein-ligand", load_binding_pocket=False)
print("Number of files in PDBBind v2015 general set: %d" % len(pdbbind_v2015_general_files))

pdbbind_v2019_PP_files = get_pdbbind_molecular_complex_files(subset="general", version="v2019", interactions="protein-protein", load_binding_pocket=False)
print("Number of files in PDBBind v2019 PP set: %d" % len(pdbbind_v2019_PP_files))

pdbbind_v2019_PN_files = get_pdbbind_molecular_complex_files(subset="general", version="v2019", interactions="protein-nucleic-acid", load_binding_pocket=False)
print("Number of files in PDBBind v2019 PN set: %d" % len(pdbbind_v2019_PN_files))

pdbbind_v2019_NL_files = get_pdbbind_molecular_complex_files(subset="general", version="v2019", interactions="nucleic-acid-ligand", load_binding_pocket=False)
print("Number of files in PDBBind v2019 NL set: %d" % len(pdbbind_v2019_NL_files))

pdbbind_v2019_refined_files = get_pdbbind_molecular_complex_files(subset="refined", version="v2019", interactions="protein-ligand", load_binding_pocket=False)
print("Number of files in PDBBind v2019 refined set: %d" % len(pdbbind_v2019_refined_files))

pdbbind_v2019_general_files = get_pdbbind_molecular_complex_files(subset="general", version="v2019", interactions="protein-ligand", load_binding_pocket=False)
print("Number of files in PDBBind v2019 general set: %d" % len(pdbbind_v2019_general_files))
