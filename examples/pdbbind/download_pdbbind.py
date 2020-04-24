# This example demonstrates how to download the raw PDBBind data 
import deepchem as dc
import logging

logging.basicConfig(level=logging.INFO)


# Download 2015 protein-ligand dataset
dc.molnet.load_function.pdbbind_datasets.download_pdbbind(version="v2015", interactions="protein-ligand")

# Download 2019 protein-protein dataset
dc.molnet.load_function.pdbbind_datasets.download_pdbbind(version="v2019", interactions="protein-protein")

# Download 2019 protein-nucleic-acid dataset
dc.molnet.load_function.pdbbind_datasets.download_pdbbind(version="v2019", interactions="protein-nucleic-acid")

# Download 2019 nucleic-acid-ligand dataset
dc.molnet.load_function.pdbbind_datasets.download_pdbbind(version="v2019", interactions="nucleic-acid-ligand")

# Download 2019 protein-ligand refined dataset
dc.molnet.load_function.pdbbind_datasets.download_pdbbind(version="v2019", interactions="protein-ligand", subset="refined")

# Download 2019 protein-ligand other dataset
dc.molnet.load_function.pdbbind_datasets.download_pdbbind(version="v2019", interactions="protein-ligand", subset="other")
