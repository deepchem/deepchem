"""
Build data frames for classification datasets with separate files for active
and inactive molecules (e.g. DUD-E and MUV).
"""
import argparse
import numpy as np
import pandas as pd
import warnings

from vs_utils.utils import write_dataframe
from vs_utils.utils.rdkit_utils import serial


def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--assay', required=1,
                      help='Assay ID.')
  parser.add_argument('--target', required=1,
                      help='Assay target.')
  parser.add_argument('-a', '--actives', required=1,
                      help='File containing actives.')
  parser.add_argument('-d', '--decoys', required=1,
                      help='File containing decoys.')
  parser.add_argument('--no-assay', action='store_false', dest='with_assay',
                      help='Do not include AID with each data point.')
  parser.add_argument('--no-target', action='store_false', dest='with_target',
                      help='Do not include target with each data point.')
  parser.add_argument('--phenotype',
                      help='Phenotype for actives in this assay.')
  parser.add_argument('-o', '--output',
                      help='Output filename.')
  parser.add_argument('-f', '--format',
                      choices=['csv', 'csv.gz', 'pkl', 'pkl.gz'],
                      default='pkl.gz',
                      help='Output file format.')
  parser.add_argument('--mols',
                      help='Filename to write unique molecules.')
  parser.add_argument('--mol-prefix',
                      help='Prefix for molecule IDs.')
  return parser.parse_args(input_args)


def get_rows(reader, outcome, phenotype=None, mol_id_prefix=None):
  """Get a row for each data point."""
  rows = []
  mol_ids = set()
  for mol in reader:
    mol_id = mol.GetProp('_Name')
    if mol_id_prefix is not None:
      mol_id = mol_id_prefix + mol_id
    row = {'mol_id': mol_id, 'outcome': outcome}
    if mol_id in mol_ids:
      warnings.warn(
          'Merging duplicated "{}" mol_id "{}"'.format(outcome, mol_id))
      continue
    mol_ids.add(mol_id)
    if phenotype is not None:
      row['phenotype'] = phenotype
    rows.append(row)
  return rows


def main(active_filename, decoy_filename, assay_id, target, with_assay_id=True,
         with_target=True, phenotype=None, output_filename=None,
         mol_id_prefix=None, output_format='.pkl.gz'):
  rows = []
  for outcome, filename in zip(['active', 'inactive'],
                               [active_filename, decoy_filename]):
    this_phenotype = phenotype
    if outcome == 'inactive' and phenotype is not None:
      this_phenotype = 'inactive'
    with serial.MolReader().open(filename) as reader:
      this_rows = get_rows(reader, outcome, this_phenotype, mol_id_prefix)
      rows.extend(this_rows)

  # create dataframe
  df = pd.DataFrame(rows)

  # sanity check for duplicate mol_ids
  assert len(np.unique(df['mol_id'])) == len(df)

  # add assay_id and target columns
  if with_assay_id:
    df.loc[:, 'assay_id'] = assay_id
  if with_target:
    df.loc[:, 'target'] = target

  if output_filename is None:
    output_filename = '{}.{}'.format(assay_id, output_format)
  print '{}\t{}\t{}\t{}'.format(assay_id, target, output_filename, len(df))
  write_dataframe(df, output_filename)

if __name__ == '__main__':
  args = parse_args()
  print args
  main(args.actives, args.decoys, args.assay, args.target, args.with_assay,
       args.with_target, args.phenotype, args.output, args.mol_prefix,
       args.format)
