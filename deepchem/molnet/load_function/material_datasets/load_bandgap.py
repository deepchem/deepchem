"""
Experimental bandgaps for inorganic crystals.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

BANDGAP_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/expt_gap.tar.gz'
BANDGAP_TASKS = ['experimental_bandgap']


class _BandgapLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, 'expt_gap.json')
    targz_file = os.path.join(self.data_dir, 'expt_gap.tar.gz')
    if not os.path.exists(dataset_file):
      if not os.path.exists(targz_file):
        dc.utils.data_utils.download_url(
            url=BANDGAP_URL, dest_dir=self.data_dir)
      dc.utils.data_utils.untargz_file(targz_file, self.data_dir)
    loader = dc.data.JsonLoader(
        tasks=self.tasks,
        feature_field="composition",
        label_field="experimental_bandgap",
        featurizer=self.featurizer)
    return loader.create_dataset(dataset_file)


def load_bandgap(
    featurizer: Union[dc.feat.Featurizer,
                      str] = dc.feat.ElementPropertyFingerprint(),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load band gap dataset.

  Contains 4604 experimentally measured band gaps for inorganic
  crystal structure compositions. In benchmark studies, random forest
  models achieved a mean average error of 0.45 eV during five-fold
  nested cross validation on this dataset.

  For more details on the dataset see [1]_. For more details
  on previous benchmarks for this dataset, see [2]_.

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

  Returns
  -------
  tasks, datasets, transformers : tuple
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
  .. [1] Zhuo, Y. et al. "Predicting the Band Gaps of Inorganic Solids by Machine Learning."
     J. Phys. Chem. Lett. (2018) DOI: 10.1021/acs.jpclett.8b00124.
  .. [2] Dunn, A. et al. "Benchmarking Materials Property Prediction Methods: The Matbench Test Set
     and Automatminer Reference Algorithm." https://arxiv.org/abs/2005.00707 (2020)

  Examples
  --------
  >>>
  >> import deepchem as dc
  >> tasks, datasets, transformers = dc.molnet.load_bandgap()
  >> train_dataset, val_dataset, test_dataset = datasets
  >> n_tasks = len(tasks)
  >> n_features = train_dataset.get_data_shape()[0]
  >> model = dc.models.MultitaskRegressor(n_tasks, n_features)

  """
  loader = _BandgapLoader(featurizer, splitter, transformers, BANDGAP_TASKS,
                          data_dir, save_dir, **kwargs)
  return loader.load_dataset('bandgap', reload)
