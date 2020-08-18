"""
Clinical Toxicity (clintox) dataset loader.
@author Caleb Geniesse
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
CLINTOX_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"


def load_clintox(featurizer='ECFP',
                 split='index',
                 reload=True,
                 data_dir=None,
                 save_dir=None,
                 **kwargs):
  """Load ClinTox dataset

  The ClinTox dataset compares drugs approved by the FDA and
  drugs that have failed clinical trials for toxicity reasons.
  The dataset includes two classification tasks for 1491 drug
  compounds with known chemical structures: 
  
  #. clinical trial toxicity (or absence of toxicity)
  #. FDA approval status.
  
  List of FDA-approved drugs are compiled from the SWEETLEAD
  database, and list of drugs that failed clinical trials for
  toxicity reasons are compiled from the Aggregate Analysis of
  ClinicalTrials.gov(AACT) database.

  Random splitting is recommended for this dataset.

  The raw data csv file contains columns below:
  
  - "smiles" - SMILES representation of the molecular structure
  - "FDA_APPROVED" - FDA approval status
  - "CT_TOX" - Clinical trial results

  References
  ----------
  .. [1] Gayvert, Kaitlyn M., Neel S. Madhukar, and Olivier Elemento.
     "A data-driven approach to predicting successes and failures of clinical
     trials."
     Cell chemical biology 23.10 (2016): 1294-1301.
  .. [2] Artemov, Artem V., et al. "Integrated deep learned transcriptomic and
     structure-based predictor of clinical trials outcomes." bioRxiv (2016): 
     095653.
  .. [3] Novick, Paul A., et al. "SWEETLEAD: an in silico database of approved
     drugs, regulated chemicals, and herbal isolates for computer-aided drug 
     discovery." PloS one 8.11 (2013): e79568.
  .. [4] Aggregate Analysis of ClincalTrials.gov (AACT) Database.
     https://www.ctti-clinicaltrials.org/aact-database
  """
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "clintox-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  dataset_file = os.path.join(data_dir, "clintox.csv.gz")
  if not os.path.exists(dataset_file):
    deepchem.utils.download_url(url=CLINTOX_URL, dest_dir=data_dir)

  logger.info("About to load clintox dataset.")
  dataset = deepchem.utils.save.load_from_disk(dataset_file)
  clintox_tasks = dataset.columns.values[1:].tolist()
  logger.info("Tasks in dataset: %s" % (clintox_tasks))
  logger.info("Number of tasks in dataset: %s" % str(len(clintox_tasks)))
  logger.info("Number of examples in dataset: %s" % str(dataset.shape[0]))
  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return clintox_tasks, all_dataset, transformers
  # Featurize clintox dataset
  logger.info("About to featurize clintox dataset.")
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
      tasks=clintox_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform clintox dataset
  if split is None:
    transformers = [deepchem.trans.BalancingTransformer(dataset=dataset)]

    logger.info("Split is None, about to transform data.")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return clintox_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'stratified': deepchem.splits.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split data with {} splitter.".format(split))
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = [deepchem.trans.BalancingTransformer(dataset=train)]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)

  return clintox_tasks, (train, valid, test), transformers
