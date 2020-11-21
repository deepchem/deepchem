"""
Clinical Toxicity (clintox) dataset loader.
@author Caleb Geniesse
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

CLINTOX_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
CLINTOX_TASKS = ['FDA_APPROVED', 'CT_TOX']


class _ClintoxLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, "clintox.csv.gz")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=CLINTOX_URL, dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)
    return loader.create_dataset(dataset_file, shard_size=8192)


def load_clintox(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
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

  Parameters
  ----------
  featurizer: Featurizer or str
    the featurizer to use for processing the data.  Alternatively you can pass
    one of the names from dc.molnet.featurizers as a shortcut.
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
  loader = _ClintoxLoader(featurizer, splitter, transformers, CLINTOX_TASKS,
                          data_dir, save_dir, **kwargs)
  return loader.load_dataset('clintox', reload)
