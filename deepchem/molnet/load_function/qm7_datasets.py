"""
qm7 dataset loader.
"""
import os
import numpy as np
import deepchem
import scipy.io
import logging

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
QM7_MAT_UTL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat"
QM7_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv"
QM7B_MAT_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat"
GDB7_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb7.tar.gz"


def load_qm7_from_mat(featurizer='CoulombMatrix',
                      split='stratified',
                      reload=True,
                      move_mean=True,
                      data_dir=None,
                      save_dir=None,
                      **kwargs):

  qm7_tasks = ["u0_atom"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "qm7-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return qm7_tasks, all_dataset, transformers

  if featurizer == 'CoulombMatrix':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM7_MAT_URL, dest_dir=data_dir)

    dataset = scipy.io.loadmat(dataset_file)
    X = dataset['X']
    y = dataset['T'].T
    w = np.ones_like(y)
    dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)
  elif featurizer == 'BPSymmetryFunctionInput':
    dataset_file = os.path.join(data_dir, "qm7.mat")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM7_MAT_URL, dest_dir=data_dir)
    dataset = scipy.io.loadmat(dataset_file)
    X = np.concatenate([np.expand_dims(dataset['Z'], 2), dataset['R']], axis=2)
    y = dataset['T'].reshape(-1, 1)  # scipy.io.loadmat puts samples on axis 1
    w = np.ones_like(y)
    dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)
  else:
    dataset_file = os.path.join(data_dir, "qm7.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM7_CSV_URL, dest_dir=data_dir)
    if featurizer == 'ECFP':
      featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()
    elif featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      img_size = kwargs.get("img_size", 80)
      featurizer = deepchem.feat.SmilesToImage(
          img_size=img_size, img_spec=img_spec)
    loader = deepchem.data.CSVLoader(
        tasks=qm7_tasks, smiles_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file)

  if split == None:
    raise ValueError()
  else:
    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'stratified':
        deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
    }

    splitter = splitters[split]
    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test)

    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)
    if reload:
      deepchem.utils.save.save_dataset_to_disk(
          save_folder, train_dataset, valid_dataset, test_dataset, transformers)

    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7b_from_mat(featurizer='CoulombMatrix',
                       split='stratified',
                       reload=True,
                       move_mean=True,
                       data_dir=None,
                       save_dir=None,
                       **kwargs):
  """Load QM7B dataset

  QM7b is an extension for the QM7 dataset with additional properties predicted
  at different levels (ZINDO, SCS, PBE0, GW). In total 14 tasks are included 
  for 7211 molecules with up to 7 heavy atoms.

  Random splitting is recommended for this dataset.

  The data file (.mat format, we recommend using `scipy.io.loadmat` 
  for python users to load this original data) contains two arrays:

  - "X" - (7211 x 23 x 23), Coulomb matrices
  - "T" - (7211 x 14), properties:

    #. Atomization energies E (PBE0, unit: kcal/mol)
    #. Excitation of maximal optimal absorption E_max (ZINDO, unit: eV)
    #. Absorption Intensity at maximal absorption I_max (ZINDO)
    #. Highest occupied molecular orbital HOMO (ZINDO, unit: eV)
    #. Lowest unoccupied molecular orbital LUMO (ZINDO, unit: eV)
    #. First excitation energy E_1st (ZINDO, unit: eV)
    #. Ionization potential IP (ZINDO, unit: eV)
    #. Electron affinity EA (ZINDO, unit: eV)
    #. Highest occupied molecular orbital HOMO (PBE0, unit: eV)
    #. Lowest unoccupied molecular orbital LUMO (PBE0, unit: eV)
    #. Highest occupied molecular orbital HOMO (GW, unit: eV)
    #. Lowest unoccupied molecular orbital LUMO (GW, unit: eV)
    #. Polarizabilities α (PBE0, unit: Å^3)
    #. Polarizabilities α (SCS, unit: Å^3)

  References
  ----------
  .. [1] Blum, Lorenz C., and Jean-Louis Reymond. "970 million druglike
     small molecules for virtual screening in the chemical universe database 
     GDB-13."
     Journal of the American Chemical Society 131.25 (2009): 8732-8733.
  .. [2] Montavon, Grégoire, et al. "Machine learning of molecular electronic
     properties in chemical compound space." New Journal of Physics 15.9 
     (2013): 095003.
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  dataset_file = os.path.join(data_dir, "qm7b.mat")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=QM7B_MAT_URL, dest_dir=data_dir)
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = deepchem.data.DiskDataset.from_numpy(X, y, w, ids=None)

  if split == None:
    raise ValueError()
  else:
    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'stratified':
        deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
    }
    splitter = splitters[split]
    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test)

    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]

    for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)

    qm7_tasks = np.arange(y.shape[1])
    return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
  """Load QM7 dataset

  QM7 is a subset of GDB-13 (a database of nearly 1 billion
  stable and synthetically accessible organic molecules)
  containing up to 7 heavy atoms C, N, O, and S. The 3D
  Cartesian coordinates of the most stable conformations and
  their atomization energies were determined using ab-initio
  density functional theory (PBE0/tier2 basis set). This dataset
  also provided Coulomb matrices as calculated in [Rupp et al.
  PRL, 2012]:

  Stratified splitting is recommended for this dataset.

  The data file (.mat format, we recommend using `scipy.io.loadmat` 
  for python users to load this original data) contains five arrays:

  - "X" - (7165 x 23 x 23), Coulomb matrices
  - "T" - (7165), atomization energies (unit: kcal/mol)
  - "P" - (5 x 1433), cross-validation splits as used in [Montavon et al. 
    NIPS, 2012]
  - "Z" - (7165 x 23), atomic charges
  - "R" - (7165 x 23 x 3), cartesian coordinate (unit: Bohr) of each atom in
    the molecules

  References
  ----------
  .. [1] Rupp, Matthias, et al. "Fast and accurate modeling of molecular
     atomization energies with machine learning." Physical review letters 
     108.5 (2012): 058301.
  .. [2] Montavon, Grégoire, et al. "Learning invariant representations of
     molecules for atomization energy prediction." Advances in Neural 
     Information Proccessing Systems. 2012.
  """
  # Featurize qm7 dataset
  logger.info("About to featurize qm7 dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR
  dataset_file = os.path.join(data_dir, "gdb7.sdf")

  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=GDB7_URL, dest_dir=data_dir)
    deepchem.utils.untargz_file(os.path.join(data_dir, 'gdb7.tar.gz'), data_dir)

  qm7_tasks = ["u0_atom"]
  if featurizer == 'CoulombMatrix':
    featurizer = deepchem.feat.CoulombMatrixEig(23)
  loader = deepchem.data.SDFLoader(
      tasks=qm7_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  if split == None:
    raise ValueError()

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)

  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers
