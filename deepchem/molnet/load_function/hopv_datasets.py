"""
HOPV dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

HOPV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hopv.tar.gz"
HOPV_TASKS = [
    'HOMO', 'LUMO', 'electrochemical_gap', 'optical_gap', 'PCE', 'V_OC', 'J_SC',
    'fill_factor'
]


class _HOPVLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, "hopv.csv")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=HOPV_URL, dest_dir=self.data_dir)
      dc.utils.data_utils.untargz_file(
          os.path.join(self.data_dir, 'hopv.tar.gz'), self.data_dir)
    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)
    return loader.create_dataset(dataset_file, shard_size=8192)


def load_hopv(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Load HOPV datasets. Does not do train/test split

  The HOPV datasets consist of the "Harvard Organic
  Photovoltaic Dataset. This dataset includes 350 small
  molecules and polymers that were utilized as p-type materials
  in OPVs. Experimental properties include: HOMO [a.u.], LUMO
  [a.u.], Electrochemical gap [a.u.], Optical gap [a.u.], Power
  conversion efficiency [%], Open circuit potential [V], Short
  circuit current density [mA/cm^2], and fill factor [%].
  Theoretical calculations in the original dataset have been
  removed (for now).

  Lopez, Steven A., et al. "The Harvard organic photovoltaic dataset." Scientific data 3.1 (2016): 1-7.

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
  """
  loader = _HOPVLoader(featurizer, splitter, transformers, HOPV_TASKS, data_dir,
                       save_dir, **kwargs)
  return loader.load_dataset('hopv', reload)
