from logging import raiseExceptions
import os
import subprocess


def system_call(command):
  """ Wrapper for system command call """
  p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
  return p.stdout.read()


def hhblits(dataset_path,
            database=None,
            data_dir=None,
            evalue=0.001,
            num_iterations=2,
            num_threads=4):
  """
  Run hhblits multisequence alignment search on a dataset. This function
  requires the hhblits binary to be installed and in the path. This function
  also requires a Hidden Markov Model reference database to be provided. Both can be
  found here: https://github.com/soedinglab/hh-suite

  The database should be in the deepchem data directory or specified as an argument.
  To set the deepchem data directory, run this command in your environment:

  export DEEPCHEM_DATA_DIR=<path to data directory>

  Example:

  >>> from deepchem.utils.sequence_utils import hhblits
  >>> hhblits('path/to/dataset.seq', database='uniprot20_2016_02', data_dir='path/to/data_dir', evalue=0.001, num_iterations=2, num_threads=4)

  The output files results.a3m amd results.hhr will be saved in the dataset directory. results.a3m is a MSA and results.hhr is a hhsuite results file.


  Parameters
  ----------
  dataset_path: str
    Path to single sequence or multiple sequence alignment (MSA) dataset. Results will be saved in this directory.
  database: str
    Name of database to search against. Note this is not the path, but the name of the database.
  data_dir: str
    Path to database directory.
  evalue: float
    E-value cutoff.
  num_iterations: int
    Number of iterations.
  num_threads: int
    Number of threads.

  """

  if data_dir is None:
    data_dir = os.environ['DEEPCHEM_DATA_DIR']
  if len(data_dir) == 0:
    raiseExceptions(
        'hhblits requires a database. Please follow the instructions here \
    to download a database: https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases'
    )

  _, dataset_file_type = os.path.splitext(dataset_path)

  save_dir = os.path.dirname(os.path.realpath(dataset_path))

  if dataset_file_type == '.fas' or '.fasta':
    command = 'hhsearch ' + \
        ' -i ' + os.path.abspath(dataset_path) + \
        ' -d ' + os.path.join(data_dir, database) + \
        ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
        ' -cpu ' + str(num_threads) + \
        ' -n ' + str(num_iterations) + \
        ' -e ' + str(evalue) + \
        ' -M first'
  if dataset_file_type == '.a3m' or '.a2m' or '.hmm':
    command = 'hhsearch ' + \
        ' -i ' + os.path.abspath(dataset_path) + \
        ' -d ' + os.path.join(data_dir, database) + \
        ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
        ' -cpu ' + str(num_threads) + \
        ' -n ' + str(num_iterations) + \
        ' -e ' + str(evalue)
  else:
    raiseExceptions('Unsupported file type')

  system_call(command)

  return os.path.join(save_dir, 'results.a3m')


def hhsearch(dataset_path,
             database=None,
             data_dir=None,
             evalue=0.001,
             num_iterations=2,
             num_threads=4):
  """
  Run hhsearch multisequence alignment search on a dataset. This function
  requires the hhblits binary to be installed and in the path. This function
  also requires a Hidden Markov Model reference database to be provided. Both can be
  found here: https://github.com/soedinglab/hh-suite

  The database should be in the deepchem data directory or specified as an argument.
  To set the deepchem data directory, run this command in your environment:

  export DEEPCHEM_DATA_DIR=<path to data directory>

  Example:

  >>> from deepchem.utils.sequence_utils import hhsearch
  >>> hhsearch('path/to/dataset.seq', database='uniprot20_2016_02', data_dir='path/to/data_dir', evalue=0.001, num_iterations=2, num_threads=4)

  The output files results.a3m amd results.hhr will be saved in the dataset directory. results.a3m is a MSA and results.hhr is a hhsuite results file.

  Parameters
  ----------
  dataset_path: str
    Path to multiple sequence alignment dataset. Results will be saved in this directory.
  database: str
    Name of database to search against. Note this is not the path, but the name of the database.
  data_dir: str
    Path to database directory.
  evalue: float
    E-value cutoff.
  num_iterations: int
    Number of iterations.
  num_threads: int
    Number of threads.

  """

  if data_dir is None:
    data_dir = os.environ['DEEPCHEM_DATA_DIR']
  if len(data_dir) == 0:
    raiseExceptions(
        'hhblits requires a database. Please follow the instructions here \
    to download a database: https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases'
    )

  _, dataset_file_type = os.path.splitext(dataset_path)

  save_dir = os.path.dirname(os.path.abspath(dataset_path))

  if dataset_file_type == '.fas' or '.fasta':
    command = 'hhsearch ' + \
        ' -i ' + os.path.abspath(dataset_path) + \
        ' -d ' + os.path.join(data_dir, database) + \
        ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
        ' -cpu ' + str(num_threads) + \
        ' -e ' + str(evalue) + \
        ' -M first'
  if dataset_file_type == '.a3m' or '.a2m' or '.hmm':
    command = 'hhsearch ' + \
        ' -i ' + os.path.abspath(dataset_path) + \
        ' -d ' + os.path.join(data_dir, database) + \
        ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
        ' -cpu ' + str(num_threads) + \
        ' -e ' + str(evalue)
  else:
    raiseExceptions('Unsupported file type')

  system_call(command)

  return os.path.join(save_dir, 'results.a3m')
