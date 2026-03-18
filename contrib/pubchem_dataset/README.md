This provides a utility for generating bioassay datasets in PubChem, similar to the pcba dataset used in the original "Massively Multitask Learning" paper by Ramsunder et al 2015. The usage is as follows:

Before starting it is recommended to first set DEEPCHEM_DATA_DIR environment variable to a directory where you have at least 66GB+30GB of storage (for all PubChem SDFs+all Bioassay CSV) available

Then download the core data we will later featurize and learn on:

```bash
python download_pubchem_ftp.py
python create_smiles_mapping.py
```

Note: On an 8-core desktop computer as of Nov 2017 it took approximately 17 hours to execute create_smiles_mapping.py (that is, to extract the smiles from all the downloaded, gzipped SDF files from PubChem)

Then, parametize the create_assay_overview.py script via setting the following options:

```bash
usage: create_assay_overview.py [-h] [-d DATASET_NAME] [-g GENE_ARG]

Deepchem dataset builder for PCBA datasets

optional arguments:
  -h, --help       show this help message and exit
  -d DATASET_NAME  Choice of dataset: pcba_128, pcba_146
  -g GENE_ARG      Name of gene to create a dataset for
```

You must select either -d pcba_146, -d pcba_2475 or -g GENE_SYMBOL.

At the end you will have a file, e.g. pcba_146.csv.gz, etc file in your DEEPCHEM_DATA_DIR ready for benchmarking

Also, please note that the pcba_146 corresponds to the following query on PubChem Bioassay Search:

10000[TotalSidCount] : 1000000000[TotalSidCount] AND 30[ActiveSidCount] : 1000000000[ActiveSidCount] AND 0[TargetCount] : 1[TargetCount] AND "NCGC"[Source Name] AND "small molecule"[filt] AND "doseresponse"[filt] 

This yields (as of Dec 2017) an additional 18 bioassays beyond the core 128 bioassays in PCBA-128

pcba_2475 corresponds to:

1[TotalSidCount] : 1000000000[TotalSidCount] AND 5[ActiveSidCount] : 10000000000[ActiveSidCount] AND 0[TargetCount] : 1[TargetCount] AND "small molecule"[filt] AND "doseresponse"[filt]