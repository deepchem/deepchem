"""
Extract JSON assay descriptions (leave assay data behind).
"""
import argparse
import glob
import gzip
import json
import os

from vs_utils.utils.public_data import PcbaJsonParser


def parse_args(input_args=None):
  """
  Parse command-line arguments.

  Parameters
  ----------
  input_args : list, optional
    Input arguments. If not provided, defaults to sys.argv[1:].
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('files', nargs='+',
                      help='Directories containing PCBA JSON files.')
  return parser.parse_args(input_args)


def main(dirs):
  for this_dir in dirs:
    print this_dir
    for filename in glob.glob(os.path.join(this_dir, '*.json.gz')):
      parser = PcbaJsonParser(filename)
      tree = parser.tree
      aid = parser.get_aid()
      try:
        del tree['PC_AssaySubmit']['data']
      except KeyError as e:
        print 'JSON is not properly formatted. Please follow NCBI FTP format.'
        raise e
      with gzip.open(os.path.join(
              this_dir, '{}-desc.json.gz'.format(aid)), 'wb') as f:
          json.dump(tree, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args.files)
