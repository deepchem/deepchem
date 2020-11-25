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
    Load Platinum Adsorption Dataset

    The dataset consist of diffrent configurations of Adsorbates (i.e N and NO)
    on Platinum surface represented as Lattice and their formation energy. There
    are 648 diffrent adsorbate configuration in this datasets given in this format

    [ax][ay][az]
    [bx][by][bz]
    [cx][cy][cz]
    [number sites]
    [site1a][site1b][site1c][site type][occupation state if active site]
    [site2a][site2b][site2c][site type][occupation state if active site]

    - ax,ay, ... are cell basis vector
    - site1a,site1b,site1c are the scaled coordinates of site 1


    Parameters
    ----------
    featurizer : Featurizer (default LCNNFeaturizer)
        the featurizer to use for processing the data. Reccomended to use
        the LCNNFeaturiser.
    splitter : Splitter (default RandomSplitter)
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data will
        be included in a single dataset.
    transformers : list of TransformerGenerators or strings. the Transformers to
        apply to the data and appropritate featuriser. Does'nt require any
        transformation for LCNN_featuriser
    reload : bool
        if True, the first call for a particular featurizer and splitter will cache
        the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir : str
        a directory to save the raw data in
    save_dir : str, optional (default None)
        a directory to save the dataset in

    References
    ----------
    .. [1] Jonathan Lym, Geun Ho G. "Lattice Convolutional Neural Network Modeling of Adsorbate
       Coverage Effects"J. Phys. Chem. C 2019, 123, 18951−18959

    Examples
    --------
    >>>
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
