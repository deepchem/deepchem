"""
Prep PDBBind molecules for processing by nnscore.
"""
import argparse
import os
from vs_utils.utils.nnscore_utils import hydrogenate_and_compute_partial_charges

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--pdbbind-dir', required=1,
                      help='Directory containing pdbbind data.')
  return parser.parse_args(input_args)

def preprocess_pdbbind(pdbbind_dir):
  """Preprocess pdbbind files for Binana."""
  assert os.path.isdir(pdbbind_dir)

  # Extract the subdirectories in pdbbind_dir
  subdirs = [d for d in os.listdir(pdbbind_dir) if
      os.path.isdir(os.path.join(pdbbind_dir, d))]

  print "About to preprocess following subdirectories:"
  print subdirs

  for count, dirname in enumerate(subdirs):
    print "Processing %d-th entry %s" % (count, dirname)
    subdir = os.path.join(pdbbind_dir, dirname)
    ligand, protein = None, None
    for molfile in os.listdir(subdir):
      if "_ligand.mol2" in molfile:
        print "Input ligand: %s" % molfile 
        ligand = molfile 
      elif "_protein.pdb" in molfile:
        print "Input protein: %s" % molfile
        protein = molfile 
    if not ligand or not protein:
      raise ValueError("Ligand or Protein missing in %s" % dirname)
    ligand_file = os.path.join(subdir, ligand)
    protein_file = os.path.join(subdir, protein)

    print "About to preprocess ligand."
    hydrogenate_and_compute_partial_charges(ligand_file, "mol2", subdir)

    print "About to preprocess protein."
    hydrogenate_and_compute_partial_charges(protein_file, "pdb", subdir)

if __name__ == '__main__':
  preprocess_pdbbind(parse_args().pdbbind_dir)
