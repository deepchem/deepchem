from Bio import Seq, SeqIO, AlignIO
import os
import subprocess
import tempfile

def system_call(command):
  """ Wrapper for system command call """
  p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
  return p.stdout.read()

def hhblits(dataset_path, dataset_file_type, database=None, data_dir=None, save_dir=None, evalue=0.001, num_iterations=2, num_threads=4):
        
  """
  Run hhblits multisequence alignment search on a dataset. This function
  requires the hhblits binary to be installed and in the path. It can be
  found here: https://github.com/soedinglab/hh-suite

  Example:

  >>> from deepchem.utils.sequence_utils import hhblits
  >>> hhblits('dataset.seq', 'fasta', 'uniprot20_2016_02', 'data_dir', 'save_dir', evalue=0.001, num_iterations=2, num_threads=4)

  results.fas is a multisequence alignment and will be saved in save_dir.
  

  Parameters
  ----------
  dataset_path: str
    Path to single sequence or multiple sequence alignment (MSA) dataset.
  dataset_file_type: str
    a3m, a2m, or FASTA file format, or HMM in hhm format
  database: str
    Name of database to search against.
  data_dir: str
    Path to database directory.
  save_dir: str
    Path to save results.
  evalue: float
    E-value cutoff.
  num_iterations: int
    Number of iterations.
  num_threads: int
    Number of threads.

  """

  if data_dir is None:
    data_dir = os.environ['DEEPCHEM_DATA_DIR']

  if save_dir is None:
    save_dir = tempfile.mkdtemp()

  if dataset_file_type == 'fasta':
    command = 'hhsearch ' + \
        ' -i ' + os.path.abspath(dataset_path) + \
        ' -d ' + os.path.join(data_dir, database) + \
        ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
        ' -cpu ' + str(num_threads) + \
        ' -n ' + str(num_iterations) + \
        ' -e ' + str(evalue) + \
        ' -M first'
  else:
    command = 'hhsearch ' + \
            ' -i ' + os.path.abspath(dataset_path) + \
            ' -d ' + os.path.join(data_dir, database) + \
            ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
            ' -cpu ' + str(num_threads) + \
            ' -n ' + str(num_iterations) + \
            ' -e ' + str(evalue)

  flag = system_call(command)

  return os.path.join(save_dir, 'results.fas')

def hhsearch(dataset_path, dataset_file_type, database=None, data_dir=None, save_dir=None, evalue=0.001, num_iterations=2, num_threads=4):
        
  """
  Run hhsearch multisequence alignment search on a dataset. This function
  requires the hhblits binary to be installed and in the path. It can be
  found here: https://github.com/soedinglab/hh-suite

  Example:
  >>> from deepchem.utils.sequence_utils import hhsearch
  >>> hhsearch('dataset.seq', 'fasta', 'uniprot20_2016_02', 'data_dir', 'save_dir', evalue=0.001, num_iterations=2, num_threads=4)
  
  results.fas is a multisequence alignment and will be saved in save_dir.

  Parameters
  ----------
  dataset_path: str
    Path to multiple sequence alignment dataset.
  dataset_file_type: str
    a2m, a3m, FASTA file format or HMM
  database: str
    Name of database to search against.
  data_dir: str
    Path to database directory.
  save_dir: str
    Path to save results.
  evalue: float
    E-value cutoff.
  num_iterations: int
    Number of iterations.
  num_threads: int
    Number of threads.

  """

  if data_dir is None:
    data_dir = os.environ['DEEPCHEM_DATA_DIR']

  if save_dir is None:
    save_dir = tempfile.mkdtemp()

  if dataset_file_type == 'fasta':
    command = 'hhsearch ' + \
        ' -i ' + os.path.abspath(dataset_path) + \
        ' -d ' + os.path.join(data_dir, database) + \
        ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
        ' -cpu ' + str(num_threads) + \
        ' -n ' + str(num_iterations) + \
        ' -e ' + str(evalue) + \
        ' -M first'
  else:
    command = 'hhsearch ' + \
            ' -i ' + os.path.abspath(dataset_path) + \
            ' -d ' + os.path.join(data_dir, database) + \
            ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
            ' -cpu ' + str(num_threads) + \
            ' -n ' + str(num_iterations) + \
            ' -e ' + str(evalue)

  flag = system_call(command)

  return os.path.join(save_dir, 'results.fas')
