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

  Returns
  -------
  results: .a3m file
    MSA file containing the results of the hhblits search.
  results: .hhr file
    hhsuite results file containing the results of the hhblits search.

  Examples
  --------
  >>> from deepchem.utils.sequence_utils import hhblits
  >>> msa_path = hhblits('test/data/example.fasta', database='example_db', data_dir='test/data/', evalue=0.001, num_iterations=2, num_threads=4)

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

  msa_path = os.path.join(save_dir, 'results.a3m')

  return msa_path


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

  Examples
  --------
  >>> from deepchem.utils.sequence_utils import hhsearch
  >>> msa_path = hhsearch('test/data/example.fasta', database='example_db', data_dir='test/data/', evalue=0.001, num_iterations=2, num_threads=4)

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

  Returns
  -------
  results: .a3m file
    MSA file containing the results of the hhblits search.
  results: .hhr file
    hhsuite results file containing the results of the hhblits search.

  """

  if data_dir is None:
    data_dir = os.environ['DEEPCHEM_DATA_DIR']
  if len(data_dir) == 0:
    raiseExceptions(
        'hhsearch requires a database. Please follow the instructions here \
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

  msa_path = os.path.join(save_dir, 'results.a3m')

  return msa_path


def MSA_to_dataset(msa_path):
  """
  Convert a multiple sequence alignment to a NumpyDataset object.
  """

  from deepchem.data.datasets import NumpyDataset  # NumpyDataset depends on utils, so imported here to prevent circular import
  from Bio import SeqIO

  with open(msa_path, 'r') as f:
    ids = []
    sequences = []
    for record in SeqIO.parse(f, 'fasta'):
      ids.append(record.id)
      seq = []
      for res in record:
        seq.append(res)
      sequences.append(seq)
    dataset = NumpyDataset(X=sequences, ids=ids)
    return dataset
