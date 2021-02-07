"""
PDBBind dataset loader.
"""
import os
import numpy as np

import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

PDBBIND_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/"
PDBBIND_TASKS = ['-logKd/Ki']


class _PDBBindLoader(_MolnetLoader):

  def __init__(self, *args, set_name: str = 'refined', **kwargs):
    super(_PDBBindLoader, self).__init__(*args, **kwargs)
    self.set_name = set_name
    if set_name == 'general':
      self.name = 'pdbbind_v2019_other_PL'  # 'general' set folder name
    else:
      self.name = 'pdbbind_v2019_refined'

  def create_dataset(self) -> Dataset:
    if self.set_name not in ['refined', 'general']:
      raise ValueError(
          "Only 'refined' and 'general' are supported for set_name.")
    filename = self.name + '.tar.gz'
    data_folder = os.path.join(self.data_dir, self.name)
    dataset_file = os.path.join(self.data_dir, filename)
    if not os.path.exists(data_folder):
      dc.utils.data_utils.download_url(
          url=PDBBIND_URL + filename, dest_dir=self.data_dir)
      dc.utils.data_utils.untargz_file(dataset_file, dest_dir=self.data_dir)

    # get pdb and sdf filenames, labels and pdbids
    protein_files, ligand_files, labels, pdbs = self._process_pdbs()

    # load and featurize each complex
    features = self.featurizer.featurize(list(zip(ligand_files, protein_files)))
    dataset = dc.data.DiskDataset.from_numpy(features, y=labels, ids=pdbs)

    return dataset

  def _process_pdbs(self) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    if self.set_name == 'general':
      data_folder = os.path.join(self.data_dir, 'v2019-other-PL')
      index_labels_file = os.path.join(data_folder,
                                       'index/INDEX_general_PL_data.2019')
    elif self.set_name == 'refined':
      data_folder = os.path.join(self.data_dir, 'refined-set')
      index_labels_file = os.path.join(data_folder,
                                       'index/INDEX_refined_data.2019')

    # Extract locations of data
    with open(index_labels_file, "r") as g:
      pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]

    protein_files = [
        os.path.join(data_folder, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs
    ]
    ligand_files = [
        os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs
    ]

    # Extract labels
    with open(index_labels_file, "r") as g:
      labels = np.array([
          # Lines have format
          # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
          # The base-10 logarithm, -log kd/pk
          float(line.split()[3]) for line in g.readlines() if line[0] != "#"
      ])

    return (protein_files, ligand_files, labels, pdbs)


def load_pdbbind(
    featurizer: dc.feat.ComplexFeaturizer,
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    set_name: str = 'refined',
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load PDBBind dataset.

  The PDBBind dataset includes experimental binding affinity data
  and structures for 4852 protein-ligand complexes from the "refined set"
  and 12800 complexes from the "general set" in PDBBind v2019.
  The refined set removes data with obvious problems
  in 3D structure, binding data, or other aspects and should therefore
  be a better starting point for docking/scoring studies. Details on
  the criteria used to construct the refined set can be found in [4]_.
  The general set does not include the refined set.

  Random splitting is recommended for this dataset.

  The raw dataset contains the columns below:

  - "ligand" - SDF of the molecular structure
  - "protein" - PDB of the protein structure
  - "CT_TOX" - Clinical trial results

  Parameters
  ----------
  featurizer: ComplexFeaturizer or str
    the complex featurizer to use for processing the data.
    Alternatively you can pass one of the names from
    dc.molnet.featurizers as a shortcut.
  splitter: Splitter or str
    the splitter to use for splitting the data into training, validation, and
    test sets.  Alternatively you can pass one of the names from
    dc.molnet.splitters as a shortcut.  If this is None, all the data
    will be included in a single dataset.
  transformers: list of TransformerGenerators or strings
    the Transformers to apply to the data.  Each one is specified by a
    TransformerGenerator or, as a shortcut, one of the names from
    dc.molnet.transformers.
  reload: bool
    if True, the first call for a particular featurizer and splitter will cache
    the datasets to disk, and subsequent calls will reload the cached datasets.
  data_dir: str
    a directory to save the raw data in
  save_dir: str
    a directory to save the dataset in
  set_name: str (default 'refined')
    Name of dataset to download. 'refined' and 'general' are supported.

  Returns
  -------
  tasks, datasets, transformers: tuple
    tasks: list
      Column names corresponding to machine learning target variables.
    datasets: tuple
      train, validation, test splits of data as
      ``deepchem.data.datasets.Dataset`` instances.
    transformers: list
      ``deepchem.trans.transformers.Transformer`` instances applied
      to dataset.

  References
  ----------
  .. [1] Liu, Z.H. et al. Acc. Chem. Res. 2017, 50, 302-309. (PDBbind v.2016)
  .. [2] Liu, Z.H. et al. Bioinformatics, 2015, 31, 405-412. (PDBbind v.2014)
  .. [3] Li, Y. et al. J. Chem. Inf. Model., 2014, 54, 1700-1716.(PDBbind v.2013)
  .. [4] Cheng, T.J. et al. J. Chem. Inf. Model., 2009, 49, 1079-1093. (PDBbind v.2009)
  .. [5] Wang, R.X. et al. J. Med. Chem., 2005, 48, 4111-4119. (Original release)
  .. [6] Wang, R.X. et al. J. Med. Chem., 2004, 47, 2977-2980. (Original release)
  """

  loader = _PDBBindLoader(
      featurizer,
      splitter,
      transformers,
      PDBBIND_TASKS,
      data_dir,
      save_dir,
      set_name=set_name,
      **kwargs)
  return loader.load_dataset(loader.name, reload)
