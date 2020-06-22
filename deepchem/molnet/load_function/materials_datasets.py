"""
Inorganic crystal dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
MB_EXPT_GAP_URL = ''
MB_EXPT_GAP_CSV_URL = ''
MB_PEROV_URL = ''
MB_PEROV_CSV_URL = ''
MB_EXPT_METAL_URL = ''
MB_EXPT_METAL_CSV_URL = ''
MP_URL = ''
MP_CSV_URL = ''


def load_expt_gap(featurizer='ChemicalFingerprint',
                  split='random',
                  reload=True,
                  move_mean=True,
                  data_dir=None,
                  save_dir=None,
                  **kwargs):
  """Load experimental band gap dataset.

  This dataset is from the following paper

  Y. Zhuo, A. Masouri Tehrani, J. Brgoch (2018) 
  Predicting the Band Gaps of Inorganic Solids by Machine Learning 
  J. Phys. Chem. Lett. 2018, 9, 7,
  1668-1673 https:doi.org/10.1021/acs.jpclett.8b00124.

  This dataset contains 4604 experimentally measured band gaps (in eV).
  """

  # Featurize dataset
  logger.info("About to featurize experimental band gap dataset.")
  ebg_tasks = ["gap expt"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "expt-gap-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return ebg_tasks, all_dataset, transformers

  if featurizer in ['ChemicalFingerprint']:
    dataset_file = os.path.join(data_dir, "expt_gap.csv")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MB_EXPT_GAP_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'expt_gap.tar.gz'), data_dir)
  else:
    dataset_file = os.path.join(data_dir, "expt_gap.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MB_EXPT_GAP_CSV_URL, dest_dir=data_dir)

  if featurizer in ['ChemicalFingerprint']:
    if featurizer == 'ChemicalFingerprint':
      featurizer = deepchem.feat.ChemicalFingerprint(data_source='matminer')
    loader = deepchem.data.CSVCompositionLoader(
        tasks=ebg_tasks, id_field="composition", featurizer=featurizer)

  dataset = loader.create_dataset(dataset_file, shard_size=1024)
  if split != 'random':
    raise ValueError()  # other splits not implemented

  splitter = deepchem.splits.RandomSplitter()
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

  return ebg_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_perovskites(featurizer='ChemicalFingerprint',
                     split='random',
                     reload=True,
                     move_mean=True,
                     data_dir=None,
                     save_dir=None,
                     **kwargs):
  """Load perovskite dataset.

  This dataset is from the following paper

  Ivano E. Castelli, David D. Landis, Kristian S. Thygesen, Søren Dahl, 
  Ib Chorkendorff, Thomas F. Jaramillo and Karsten W. Jacobsen (2012) 
  New cubic perovskites for one- and two-photon water splitting using the
  computational materials repository. 
  Energy Environ. Sci., 2012,5, 9034-9043 
  https://doi.org/10.1039/C2EE22341D

  This dataset contains 18928 perovskite heats of formation (in eV)
  calculated with PBE GGA-DFT.
  """

  logger.info("About to featurize perovskite dataset.")
  perovskite_tasks = ["e_form"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "perovskite-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return perovskite_tasks, all_dataset, transformers

  if featurizer in ['ChemicalFingerprint']:
    dataset_file = os.path.join(data_dir, "perovskite.csv")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MB_PEROV_CSV_URL, dest_dir=data_dir)
  else:
    dataset_file = os.path.join(data_dir, "perovskite.json")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MB_PEROV_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'perovskite.tar.gz'), data_dir)

  if featurizer in ['ChemicalFingerprint']:
    if featurizer == 'ChemicalFingerprint':
      featurizer = deepchem.feat.ChemicalFingerprint(data_source='matminer')
    loader = deepchem.data.CSVCompositionLoader(
        tasks=perovskite_tasks, id_field="composition", featurizer=featurizer)

  if featurizer in ['SineCoulombMatrix', 'StructureGraph']:
    if featurizer == 'SineCoulombMatrix':
      featurizer = deepchem.feat.SineCoulombMatrix(5)
    elif featurizer == 'StructureGraph':
      featurizer = deepchem.feat.StructureGraph()
    loader = deepchem.data.JsonStructureLoader(
        tasks=perovskite_tasks, id_field="structure", featurizer=featurizer)

  dataset = loader.create_dataset(dataset_file, shard_size=8192)
  if split != 'random':
    raise ValueError()  # other splits not implemented

  splitter = deepchem.splits.RandomSplitter()
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

  return perovskite_tasks, (train_dataset, valid_dataset,
                            test_dataset), transformers


def load_expt_metal(featurizer='ChemicalFingerprint',
                    split='random',
                    reload=True,
                    move_mean=True,
                    data_dir=None,
                    save_dir=None,
                    **kwargs):
  """Load experimental metallicity dataset.

  This dataset is from the following paper

  Y. Zhuo, A. Masouri Tehrani, J. Brgoch (2018) 
  Predicting the Band Gaps of Inorganic Solids by Machine Learning 
  J. Phys. Chem. Lett. 2018, 9, 7,
  1668-1673 https:doi.org/10.1021/acs.jpclett.8b00124.

  This dataset contains 4921 metallic and nonmetallic compounds.
  """

  # Featurize dataset
  logger.info("About to featurize experimental metallicity dataset.")
  metal_tasks = ["is_metal"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "expt-is-metal-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return metal_tasks, all_dataset, transformers

  if featurizer in ['ChemicalFingerprint']:
    dataset_file = os.path.join(data_dir, "expt_is_metal.csv")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MB_EXPT_METAL_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'expt_is_metal.tar.gz'), data_dir)
  else:
    dataset_file = os.path.join(data_dir, "expt_is_metal.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MB_EXPT_METAL_CSV_URL, dest_dir=data_dir)

  if featurizer in ['ChemicalFingerprint']:
    if featurizer == 'ChemicalFingerprint':
      featurizer = deepchem.feat.ChemicalFingerprint(data_source='matminer')
    loader = deepchem.data.CSVCompositionLoader(
        tasks=metal_tasks, id_field="composition", featurizer=featurizer)

  dataset = loader.create_dataset(dataset_file, shard_size=1024)
  if split != 'random':
    raise ValueError()  # other splits not implemented

  splitter = deepchem.splits.RandomSplitter()
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

  return metal_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_materials_project(featurizer='ChemicalFingerprint',
                           split='random',
                           reload=True,
                           move_mean=True,
                           data_dir=None,
                           save_dir=None,
                           **kwargs):
  """Load materials project dataset.

  The citation for this dataset is

  A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S.
  Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal
  contributions) The Materials Project: A materials genome approach to
  accelerating materials innovation APL Materials, 2013, 1(1), 011002.
  doi:10.1063/1.4812323

  This dataset contains 132752 inorganic crystal structures and their
  heats of formation (in eV) calculated with PBE GGA-DFT.
  """

  logger.info("About to featurize materials project dataset.")
  mp_tasks = ["e_form"]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "materials-project-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return mp_tasks, all_dataset, transformers

  if featurizer in ['ChemicalFingerprint']:
    dataset_file = os.path.join(data_dir, "mp.csv")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MP_CSV_URL, dest_dir=data_dir)
  else:
    dataset_file = os.path.join(data_dir, "mp.json")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=MP_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(os.path.join(data_dir, 'mp.tar.gz'), data_dir)

  if featurizer in ['ChemicalFingerprint']:
    if featurizer == 'ChemicalFingerprint':
      featurizer = deepchem.feat.ChemicalFingerprint(data_source='matminer')
    loader = deepchem.data.CSVCompositionLoader(
        tasks=mp_tasks, id_field="composition", featurizer=featurizer)

  if featurizer in ['SineCoulombMatrix', 'StructureGraph']:
    if featurizer == 'SineCoulombMatrix':
      featurizer = deepchem.feat.SineCoulombMatrix(444)
    elif featurizer == 'StructureGraph':
      featurizer = deepchem.feat.StructureGraph()
    loader = deepchem.data.JsonStructureLoader(
        tasks=mp_tasks, id_field="structure", featurizer=featurizer)

  dataset = loader.create_dataset(dataset_file, shard_size=8192)
  if split != 'random':
    raise ValueError()  # other splits not implemented

  splitter = deepchem.splits.RandomSplitter()
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

  return mp_tasks, (train_dataset, valid_dataset, test_dataset), transformers
