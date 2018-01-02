import pandas as pd
import os
from rdkit import Chem
import time
import gzip
import pickle
import deepchem


def main():
  print("Processing PubChem FTP Download")

  data_dir = deepchem.utils.get_data_dir()
  sdf_dir = os.path.join(data_dir, "SDF")

  compound_read_count = 0
  keys = list()
  values = list()
  overall_start = time.time()

  all_paths = list()

  for path, dirs, filenames in os.walk(sdf_dir):
    for filename in filenames:

      # RDKit consistently hangs when trying to read this file
      if "102125001_102150000" in filename:
        continue

      file_path = os.path.join(sdf_dir, filename)
      all_paths.append(file_path)

  all_paths.sort()

  for filepath in all_paths:

    print("Processing: {0}".format(filepath))
    start = time.time()

    with gzip.open(filepath, 'rb') as myfile:
      suppl = Chem.ForwardSDMolSupplier(myfile)
      for mol in suppl:
        if mol is None: continue
        cid = mol.GetProp("PUBCHEM_COMPOUND_CID")
        try:
          smiles = Chem.MolToSmiles(mol)
          keys.append(int(cid))
          values.append(smiles)
        except Exception:

          continue
      end = time.time()

    print("Processed file, processed thru compound number: {0} in {1} seconds".
          format(compound_read_count, end - start))
    compound_read_count = compound_read_count + 1

  overall_end = time.time()
  secs_elapsed = overall_end - overall_start
  print("Parsed all smiles in: {0} seconds, or {1} minutes, or {2} hours".
        format(secs_elapsed, secs_elapsed / 60, secs_elapsed / 3600))
  print("Total length of: {}".format(len(keys)))
  with open(os.path.join(data_dir, "/pubchemsmiles_tuple.pickle"), "wb") as f:
    pickle.dump((keys, values), f)
  print("Done")
  overall_end = time.time()
  secs_elapsed = overall_end - overall_start
  print("Sorted and saved smiles in: {0} seconds, or {1} minutes, or {2} hours".
        format(secs_elapsed, secs_elapsed / 60, secs_elapsed / 3600))


if __name__ == '__main__':
  main()
