"""
Extract target mol_id from PCBA JSON files.
"""
import argparse
import glob
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
  parser.add_argument('dirs', nargs='+',
                      help='Directories containing PCBA JSON files.')
  parser.add_argument('-o', '--output', required=1,
                      help='Output filename.')
  return parser.parse_args(input_args)


def main(dirs, output_filename):
  targets = []
  aids = []
  for this_dir in dirs:
    print this_dir
    for filename in glob.glob(os.path.join(this_dir, '*.json.gz')):
      parser = PcbaJsonParser(filename)
      aid = parser.get_aid()
      target = parser.get_target()
      if target is None or len(target) > 1:
        continue
      try:
        mol_id = target[0]['mol_id']
      except KeyError:
        print '\tAID {} target has no mol_id'.format(aid)
        continue
      print '\tAID {} => {}'.format(aid, mol_id)
      targets.append(mol_id)
      aids.append(aid)
  print 'Found {} targets'.format(len(aids))
  with open(output_filename, 'wb') as f:
    for aid, target in zip(aids, targets):
      f.write('{}\t{}\n'.format(aid, target))

if __name__ == '__main__':
  args = parse_args()
  main(args.dirs, args.output)
