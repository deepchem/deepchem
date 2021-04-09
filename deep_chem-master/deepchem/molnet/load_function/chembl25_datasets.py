"""
ChEMBL dataset loader, for training ChemNet
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

CHEMBL25_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/chembl_25.csv.gz"
CHEMBL25_TASKS = [
    "MolWt", "HeavyAtomMolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "HeavyAtomCount", "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors",
    "NumHeteroatoms", "NumRotatableBonds", "NumRadicalElectrons",
    "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "NumAromaticCarbocycles", "NumSaturatedCarbocycles",
    "NumAliphaticCarbocycles", "NumAromaticHeterocycles",
    "NumSaturatedHeterocycles", "NumAliphaticHeterocycles", "PEOE_VSA1",
    "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11",
    "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "SMR_VSA1", "SMR_VSA2",
    "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8",
    "SMR_VSA9", "SMR_VSA10", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3",
    "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
    "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "EState_VSA1",
    "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6",
    "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11",
    "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5",
    "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "VSA_EState10",
    "BalabanJ", "BertzCT", "Ipc", "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha",
    "Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n", "Chi0v",
    "Chi1v", "Chi2v", "Chi3v", "Chi4v"
]


class _Chembl25Loader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, "chembl_25.csv.gz")
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=CHEMBL25_URL, dest_dir=self.data_dir)
    loader = dc.data.CSVLoader(
        tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)
    return loader.create_dataset(dataset_file, shard_size=8192)


def load_chembl25(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """Loads the ChEMBL25 dataset, featurizes it, and does a split.

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
  loader = _Chembl25Loader(featurizer, splitter, transformers, CHEMBL25_TASKS,
                           data_dir, save_dir, **kwargs)
  return loader.load_dataset('chembl25', reload)
