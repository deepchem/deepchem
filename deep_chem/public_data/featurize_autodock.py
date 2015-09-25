"""
Featurize structures output by autodock.
"""
import argparse
from vs_utils.utils.nnscore_pdb import MultiStructure

def featurize_autodock(pdb_filename, pdbqt_filename):
  multi = MultiStructure()
  multi.load_from_files(pdb_filename, pdbqt_filename)

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--pdb', required=1,
                      help='Input PDB file.')
  parser.add_argument('--pdbqt', required=1,
                      help='Input PDBQT file.')
  return parser.parse_args(input_args)

if __name__ == '__main__':
  args = parse_args()
  featurize_autodock(args.pdb, args.pdbqt)
