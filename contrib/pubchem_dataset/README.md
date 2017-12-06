This provides a utility for generating bioassay datasets in PubChem, similar to the PCBA-128 dataset used in the original "Massively Multitask Learning" paper by Ramsunder et al 2015. The usage is as follows:

Before starting it is recommended to first set DEEPCHEM_DATA_DIR environment variable to a directory where you have at least 66GB+30GB of storage (for all PubChem SDFs+all Bioassay CSV) available

Then download the core data we will later featurize and learn on:

```bash
python download_pubchem_ftp.py
python create_smiles_mapping.py
```

Note: On an 8-core desktop computer as of Nov 2017 it took approximately 17 hours to execute create_smiles_mapping.py (that is, to extract the smiles from all the downloaded, gzipped SDF files from PubChem)

Then, parametize the create_assay_overview.py script via setting the following variables. Only one boolean should be set true. 
If parse_128_only is set it will create a summary dataset based on the original 128 bioassays. 
If parse_all_ncgc is set it will create a summary dataset based on all NCGC assays available in PubChem as of Nov 2017
The gene_symbol only needs to be set if parse_selected_gene is true. If so, it will build a results table focused on assays relevant to this gene.

```python
  parse_128_only = False
  parse_all_ncgc = False
  parse_selected_gene = True
  gene_symbol = "PPARG"
```

Then run:
```bash
python create_assay_overview.py
```

At the end you will have a pcba.csv.gz file in your DEEPCHEM_DATA_DIR ready for benchmarking
