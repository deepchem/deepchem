# Setting up PDBBind example

First, ensure that you are in the correct environment, and then download the pdbbind reference database:
```bash
source activate deepchem
(deepchem) user@server:~/deepchem/examples/pdbbind$ ./get_pdbbind.sh
```

You will see a large number of directories created with PDB (protein) and SDF (ligand) files, for instance:
```bash
v2015/3d1g/3d1g_ligand.mol2
v2015/3d1g/3d1g_pocket.pdb
v2015/3d1g/3d1g_protein.pdb
v2015/3d1g/3d1g_ligand.sdf
```

Next, you can train the Random Forest-based classifier via:
```bash
(deepchem) ubuntu@ip-172-31-12-186:~/deepchem/examples/pdbbind$ python pdbbind_rf.py 
```

TODO: Add notes about how to expand the v2015 dataset, e.g. to add protein 14HR as associated with NF2 to the analysis
