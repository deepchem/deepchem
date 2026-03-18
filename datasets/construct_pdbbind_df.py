"""
Contains methods for generating a pdbbind dataset mapping
  complexes (protein + ligand) to experimental binding measurement.
"""
from __future__ import print_function
import pickle
import os
import pandas as pd
from rdkit import Chem
from glob import glob
import re
from sklearn.externals import joblib


def extract_labels(pdbbind_label_file):
  """Extract labels from pdbbind label file."""
  assert os.path.isfile(pdbbind_label_file)
  labels = {}
  with open(pdbbind_label_file) as f:
    content = f.readlines()
    for line in content:
      if line[0] == "#":
        continue
      line = line.split()
      # lines in the label file have format
      # PDB-code Resolution Release-Year -logKd Kd reference ligand-name
      #print line[0], line[3]
      labels[line[0]] = line[3]
  return labels

def construct_df(pdb_stem_directory, pdbbind_label_file, pdbbind_df_joblib):
  """
  Takes as input a stem directory containing subdirectories with ligand
    and protein pdb/mol2 files, a pdbbind_label_file containing binding
    assay data for the co-crystallized ligand in each pdb file,
    and a pdbbind_df_pkl to which will be saved a pandas DataFrame
    where each row contains a pdb_id, smiles string, unique complex id,
    ligand pdb as a list of strings per line in file, protein pdb as a list
    of strings per line in file, ligand mol2 as a list of strings per line in
    mol2 file, and a "label" containing the experimental measurement.
  """
  labels = extract_labels(pdbbind_label_file)
  df_rows = []
  os.chdir(pdb_stem_directory)
  pdb_directories = [pdb.replace('/', '') for pdb in glob('*/')]

  for pdb_dir in pdb_directories:
    print("About to extract ligand and protein input files")
    pdb_id = os.path.basename(pdb_dir)
    ligand_pdb = None
    protein_pdb = None
    for f in os.listdir(pdb_dir):
      if re.search("_ligand_hyd.pdb$", f):
        ligand_pdb = f
      elif re.search("_protein_hyd.pdb$", f):
        protein_pdb = f
      elif re.search("_ligand.mol2$", f):
        ligand_mol2 = f

    print("Extracted Input Files:")
    print (ligand_pdb, protein_pdb, ligand_mol2)
    if not ligand_pdb or not protein_pdb or not ligand_mol2:
      raise ValueError("Required files not present for %s" % pdb_dir)
    ligand_pdb_path = os.path.join(pdb_dir, ligand_pdb)
    protein_pdb_path = os.path.join(pdb_dir, protein_pdb)
    ligand_mol2_path = os.path.join(pdb_dir, ligand_mol2)

    with open(protein_pdb_path, "rb") as f:
      protein_pdb_lines = f.readlines()

    with open(ligand_pdb_path, "rb") as f:
      ligand_pdb_lines = f.readlines()

    try:
      with open(ligand_mol2_path, "rb") as f:
        ligand_mol2_lines = f.readlines()
    except:
      ligand_mol2_lines = []

    print("About to compute ligand smiles string.")
    ligand_mol = Chem.MolFromPDBFile(ligand_pdb_path)
    if ligand_mol is None:
      continue
    smiles = Chem.MolToSmiles(ligand_mol)
    complex_id = "%s%s" % (pdb_id, smiles)
    label = labels[pdb_id]
    df_rows.append([pdb_id, smiles, complex_id, protein_pdb_lines,
                    ligand_pdb_lines, ligand_mol2_lines, label])

  pdbbind_df = pd.DataFrame(df_rows, columns=('pdb_id', 'smiles', 'complex_id',
                                              'protein_pdb', 'ligand_pdb',
                                              'ligand_mol2', 'label'))

  joblib.dump(pdbbind_df, pdbbind_df_joblib)
