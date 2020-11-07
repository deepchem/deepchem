"""
Platinum Adsorbtion structure for N and NO along with their formation energies
"""
import numpy as np
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.feat.material_featurizers.lcnn_featurizer import LCNNFeaturizer
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

PLATINUM_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/platinum_adsorption.tar.gz"
PLATINUM_TASKS = ["Formation Energy"]
PRIMITIVE_CELL_TEMPLATE = """#Primitive Cell
2.81852800e+00  0.00000000e+00  0.00000000e+00 T
-1.40926400e+00  2.44091700e+00  0.00000000e+00 T
0.00000000e+00  0.00000000e+00  2.55082550e+01 F
1 1
1 0 2
6
0.666670000000  0.333330000000  0.090220999986 S1
0.333330000000  0.666670000000  0.180439359180 S1
0.000000000000  0.000000000000  0.270657718374 S1
0.666670000000  0.333330000000  0.360876077568 S1
0.333330000000  0.666670000000  0.451094436762 S1
0.000000000000  0.000000000000  0.496569911270 A1"""


class _PtAdsorptionLoader(_MolnetLoader):

  def create_dataset(self) -> Dataset:
    dataset_file = os.path.join(self.data_dir, 'Platinum_Adsorption.json')
    if not os.path.exists(dataset_file):
      dc.utils.data_utils.download_url(url=PLATINUM_URL, dest_dir=self.data_dir)
      dc.utils.data_utils.untargz_file(
          os.path.join(self.data_dir, 'platinum_adsorption.tar.gz'),
          self.data_dir)
    loader = dc.data.JsonLoader(
        tasks=PLATINUM_TASKS,
        feature_field="Structures",
        label_field="Formation Energy",
        featurizer=self.featurizer,
        **self.args)
    return loader.create_dataset(dataset_file)


def load_Platinum_Adsorption(
    featurizer: Union[dc.feat.Featurizer, str] = LCNNFeaturizer(
        np.around(6.00), PRIMITIVE_CELL_TEMPLATE),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = [],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
  """
    Load mydataset.
    Contains 
    Parameters
    ----------
    featurizer : Featurizer (default LCNNFeaturizer)
        A featurizer that inherits from deepchem.feat.Featurizer.
    transformers : List[]
    Does'nt require any transformation
    splitter : Splitter (default RandomSplitter)
        A splitter that inherits from deepchem.splits.splitters.Splitter.
    reload : bool (default True)
        Try to reload dataset from disk if already downloaded. Save to disk
        after featurizing.
    data_dir : str, optional (default None)
        Path to datasets.
    save_dir : str, optional (default None)
        Path to featurized datasets.
    featurizer_kwargs : dict
        Specify parameters to featurizer, e.g. {"cutoff": 6.00}
    splitter_kwargs : dict
        Specify parameters to splitter, e.g. {"seed": 42}
    transformer_kwargs : dict
        Maps transformer names to constructor arguments, e.g.
        {"BalancingTransformer": {"transform_x":True, "transform_y":False}}
    **kwargs : additional optional arguments.
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
    MLA style references for this dataset. The example is like this.
    Last, First et al. "Article title." Journal name, vol. #, no. #, year, pp. page range, DOI.
    ...[1] Lym, J et al. "Lattice Convolutional Neural Network Modeling of Adsorbate
            Coverage Effects"J. Phys. Chem. C 2019, 123, 18951−18959
    Examples
    --------
    >> import deepchem as dc
    >> feat_args = {"cutoff": np.around(6.00, 2), "input_file_path": os.path.join(data_path,'input.in') }

    >> tasks, datasets, transformers = load_Platinum_Adsorption(
        reload=True,
        data_dir=data_path,
        save_dir=data_path,
        featurizer_kwargs=feat_args)
    >> train_dataset, val_dataset, test_dataset = datasets
    """

  loader = _PtAdsorptionLoader(featurizer, splitter, transformers,
                               PLATINUM_TASKS, data_dir, save_dir, **kwargs)
  return loader.load_dataset('LCNN_feat', reload)
