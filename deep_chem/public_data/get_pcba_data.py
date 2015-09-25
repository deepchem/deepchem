"""
Extract PCBA data.

One dataframe is created for each file. The AID and target are associated with
each data point, so routing can be done on a per-point basis using either
field.

Configuration File
------------------
The configuration file is a CSV file whose column headers correspond to columns
that will appear in the saved dataframes. The file must contain an 'aid' column
listing the PCBA assay IDs (AIDs) from which data will be extracted.

Here's an example configuration file:

> aid,target,potency,hill_slope
> 998,757912,Potency,Fit_HillSlope

Running the script with this configuration file will generate a output file
'aid998-gi757912-data.pkl.gz' containing a dataframe with columns ['aid',
'target', 'potency', 'hill_slope', 'efficacy', 'phenotype', 'sid', 'outcome'].

The 'potency' and 'hill_slope' columns will be populated from the 'Potency' and
'Fit_Hillslope' columns in the original data, respectively. The 'aid' and
'target' fields do not match columns in the assay data, so they are considered
constants and will be the same for each row of the dataframe.

Columns are added for fields that are standard for PCBA data, such as a column
to track SIDs ('sid') and categorical activity outcomes ('outcome').
Additionally, columns are added when commonly-occurring fields are recognized
(to simplify writing the configuration file). In this example, 'phenotype' and
'efficacy' columns are added to track the commonly-occurring 'Phenotype' and
'Efficacy' fields.
"""
import argparse
import glob
import numpy as np
import os
import pandas as pd
import warnings

from vs_utils.utils import write_dataframe
from vs_utils.utils.public_data import PcbaDataExtractor, read_sid_cid_map

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "BSD 3-clause"


def parse_args(input_args=None):
  """
  Parse command-line arguments.

  Parameters
  ----------
  input_args : list, optional
    Input arguments. If not provided, defaults to sys.argv[1:].
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('dirs', nargs='+',
                      help='Directories containing PCBA JSON files.')
  parser.add_argument('-c', '--config', required=True,
                      help='Configuration file containing assay annotations.')
  parser.add_argument('-m', '--map',
                      help='SID->CID map filename.')
  parser.add_argument('-s', '--summary',
                      help='Filename for summary information (.csv.gz).')
  parser.add_argument('--no-aid', action='store_false', dest='with_aid',
                      help='Do not include AID with each data point.')
  parser.add_argument('--no-target', action='store_false', dest='with_target',
                      help='Do not include target with each data point.')
  parser.add_argument('--phenotype', action='store_true',
                      help='Require compound-level phenotype data.')
  parser.add_argument('--prefix', default='CID',
                      help='Prefix for molecule IDs.')
  parser.add_argument('-f', '--format',
                      choices=['csv', 'csv.gz', 'pkl', 'pkl.gz'],
                      default='pkl.gz',
                      help='Output file format.')
  return parser.parse_args(input_args)


def main(dirs, config_filename, map_filename=None, summary_filename=None,
         with_aid=True, with_target=True, phenotype=False, id_prefix='CID',
         output_format='.pkl.gz'):
  aids = set()
  targets = set()
  total = 0
  config = pd.read_csv(config_filename)
  summary = []
  sid_cid = None
  if map_filename is not None:
    sid_cid = read_sid_cid_map(map_filename)
  if 'aid' not in config.columns:
    raise ValueError('Configuration file must contain "aid" column.')
  assert len(config) == len(pd.unique(config['aid']))
  for this_dir in dirs:
    for filename in glob.glob(os.path.join(this_dir, '*.json.gz')):

      # get AID from filename so we only have to load relevant assays
      aid = int(os.path.basename(filename).split('.')[0])
      if aid not in config['aid'].values:
        continue

      # get configuration for this AID
      this_config = config[config['aid'] == aid].iloc[0]
      if not with_aid and 'aid' in this_config:
        del this_config['aid']
      if not with_target and 'target' in this_config:
        del this_config['target']

      # get data
      try:
        extractor = PcbaDataExtractor(filename, this_config, with_aid=with_aid)
      except NotImplementedError as e:
        warnings.warn(e.message)
        continue
      if phenotype and 'phenotype' not in extractor.config:
        warnings.warn('{} has no phenotype'.format(aid))
        continue
      assert aid == extractor.parser.get_aid()  # sanity check for AID match
      aids.add(aid)
      target = extractor.config.get('target')
      targets.add(target)
      data = extractor.get_data(sid_cid=sid_cid)
      total += len(data)

      # add generic molecule ID column
      if id_prefix == 'CID':
        col = 'cid'
      elif id_prefix == 'SID':
        col = 'sid'
      else:
        raise NotImplementedError('Unrecognized ID prefix "{}"'.format(
            id_prefix))
      ids = []
      for i, mol_id in enumerate(data[col]):
        try:
          ids.append(id_prefix + str(int(mol_id)))
        except (TypeError, ValueError):
          warnings.warn('No ID for the following row:\n{}'.format(data.loc[i]))
          ids.append(None)  # can be found with pd.isnull

      # skip this assay if there are no valid IDs
      if np.all(pd.isnull(ids)):
        warnings.warn('No valid IDs for AID {}. Skipping.'.format(aid))
        continue
      data.loc[:, 'mol_id'] = pd.Series(ids, index=data.index)

      # add generic assay ID column
      assay_id = 'PCBA-' + str(aid)
      if with_aid:
        data.loc[:, 'assay_id'] = assay_id

      # save dataframe
      output_filename = '{}.{}'.format(assay_id, output_format)
      print '{}\t{}\t{}\t{}'.format(aid, target, output_filename, len(data))
      write_dataframe(data, output_filename)
      summary.append({'aid': aid, 'target': target,
                      'filename': output_filename, 'size': len(data)})

  # make sure we found everything
  missing = set(config['aid']).difference(aids)
  if len(missing):
    warnings.warn('Missed AIDs {}'.format(missing))

  # save a summary
  summary = pd.DataFrame(summary)
  if summary_filename is not None:
    write_dataframe(summary, summary_filename)
  warnings.warn('Found {} assays for {} targets ({} total data points)'.format(
      len(aids), len(targets), total))

if __name__ == '__main__':
  args = parse_args()
  main(args.dirs, args.config, args.map, args.summary, args.with_aid,
       args.with_target, args.phenotype, args.prefix, args.format)
