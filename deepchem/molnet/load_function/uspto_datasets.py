"""
Loads synthetic reaction datasets from USPTO.

This file contains loaders for synthetic reaction datasets from the US Patent Office. http://nextmovesoftware.com/blog/2014/02/27/unleashing-over-a-million-reactions-into-the-wild/.
"""
import os
import logging
from typing import List, Optional, Tuple, Union

import deepchem as dc
from deepchem.data import Dataset
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader

try:
  from transformers import RobertaTokenizerFast
  from deepchem.feat.reaction_featurizer import RxnFeaturizer
except ModuleNotFoundError:
  pass

logger = logging.getLogger(__name__)

DEFAULT_DIR = dc.utils.data_utils.get_data_dir()

USPTO_MIT_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv"
USPTO_STEREO_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_STEREO.csv"
USPTO_50K_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv"
USPTO_FULL_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_FULL.csv"

USPTO_TASK: List[str] = []


class _USPTOLoader(_MolnetLoader):

  def __init__(self, *args, subset: str, sep_reagent: bool, **kwargs):
    super(_USPTOLoader, self).__init__(*args, **kwargs)
    self.subset = subset
    self.sep_reagent = sep_reagent
    self.name = 'USPTO_' + subset

  def create_dataset(self) -> Dataset:
    if self.subset not in ['MIT', 'STEREO', '50K', 'FULL']:
      raise ValueError("Valid Subset names are MIT, STEREO and 50K.")

    if self.subset == 'MIT':
      dataset_url = USPTO_MIT_URL

    if self.subset == 'STEREO':
      dataset_url = USPTO_STEREO_URL

    if self.subset == '50K':
      dataset_url = USPTO_50K_URL

    if self.subset == 'FULL':
      dataset_url = USPTO_FULL_URL
      if self.splitter == 'SpecifiedSplitter':
        raise ValueError(
            "There is no pre computed split for the full dataset, use a custom split instead!"
        )

    dataset_file = os.path.join(self.data_dir, self.name + '.csv')

    if not os.path.exists(dataset_file):
      logger.info("Downloading dataset...")
      dc.utils.data_utils.download_url(url=dataset_url, dest_dir=self.data_dir)
      logger.info("Dataset download complete.")

    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="reactions", featurizer=self.featurizer)

    return loader.create_dataset(dataset_file, shard_size=8192)


def load_uspto(
    featurizer: Union[dc.feat.Featurizer, str] = "RxnFeaturizer",
    splitter: Union[dc.splits.Splitter, str, None] = None,
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    subset: str = "MIT",
    sep_reagent: bool = True,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load USPTO Datasets.

  The USPTO dataset consists of over 1.8 Million organic chemical reactions
  extracted from US patents and patent applications. The dataset contains the
  reactions in the form of reaction SMILES, which have the general format:
  reactant>reagent>product.

  Molnet provides ability to load subsets of the USPTO dataset namely MIT,
  STEREO and 50K. The MIT dataset contains around 479K reactions, curated by
  jin et al. The STEREO dataset contains around 1 Million Reactions, it does
  not have duplicates and the reactions include stereochemical information.
  The 50K dataset contatins 50,000 reactions and is the benchmark for
  retrosynthesis predictions. The reactions are additionally classified into 10
  reaction classes. The canonicalized version of the dataset used by the loader
  is the same as that used by Somnath et. al.

  The loader uses the SpecifiedSplitter to use the same splits as specified
  by Schwaller et. al and Dai et. al. Custom splitters could also be used. There
  is a toggle in the loader to skip the source/target transformation needed for
  seq2seq tasks. There is an additional toggle to load the dataset with the
  reagents and reactants separated or mixed. This alters the entries in source
  by replacing the '>' with '.' , effectively loading them as an unified
  SMILES string.

  Parameters
  ----------
  featurizer: Featurizer or str
    the featurizer to use for processing the data.  Alternatively you can pass
    one of the names from dc.molnet.featurizers as a shortcut.
  splitter: Splitter or str
    the splitter to use for splitting the data into training, validation, and
    test sets.  Alternatively you can pass one of the names from
    dc.molnet.splitters as a shortcut. If this is None, all the data
    will be included in a single dataset.
  transformers: list of TransformerGenerators or strings
    the Transformers to apply to the data. Each one is specified by a
    TransformerGenerator or, as a shortcut, one of the names from
    dc.molnet.transformers.
  reload: bool
    if True, the first call for a particular featurizer and splitter will cache
    the datasets to disk, and subsequent calls will reload the cached datasets.
  data_dir: str
    a directory to save the raw data in
  save_dir: str
    a directory to save the dataset in
  subset: str (default 'MIT')
    Subset of dataset to download. 'FULL', 'MIT', 'STEREO', and '50K' are supported.
  sep_reagent: bool (default True)
    Toggle to load dataset with reactants and reagents either separated or mixed.
  skip_transform: bool (default True)
    Toggle to skip the source/target transformation.

  Returns
  -------
  tasks, datasets, transformers: tuple
    tasks : list
      Column names corresponding to machine learning target variables.
    datasets : tuple
      train, validation, test splits of data as
      ``deepchem.data.datasets.Dataset`` instances.
    transformers : list
      ``deepchem.trans.transformers.Transformer`` instances applied
      to dataset.

  References
  ----------
  .. [1] Lowe, D. Chemical reactions from US patents (1976-Sep2016)
        (Version 1). figshare (2017). https://doi.org/10.6084/m9.figshare.5104873.v1
  .. [2] Somnath, Vignesh Ram, et al. "Learning graph models for retrosynthesis
         prediction." arXiv preprint arXiv:2006.07038 (2020).
  .. [3] Schwaller, Philippe, et al. "Molecular transformer: a model for
         uncertainty-calibrated chemical reaction prediction."
         ACS central science 5.9 (2019): 1572-1583.
  .. [4] Dai, Hanjun, et al. "Retrosynthesis prediction with conditional
         graph logic network." arXiv preprint arXiv:2001.01408 (2020).
  """

  tokenizer = RobertaTokenizerFast.from_pretrained(
      "seyonec/PubChem10M_SMILES_BPE_450k")

  if featurizer == "plain":
    featurizer = dc.feat.DummyFeaturizer()
  else:
    featurizer = RxnFeaturizer(tokenizer, sep_reagent=sep_reagent)

  loader = _USPTOLoader(
      featurizer,
      splitter,
      transformers,
      USPTO_TASK,
      data_dir,
      save_dir,
      subset=subset,
      sep_reagent=sep_reagent,
      **kwargs)
  return loader.load_dataset(loader.name, reload)
