"""
Platinum Adsorbtion structure for N and NO along with their formation energies
"""
import numpy as np
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

PLATINUM_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Platinum_adsorption.tar.gz"
PLATINUM_TASKS = ["Formation Energy"]
PRIMITIVE_CELL = {
    "lattice": [[2.818528, 0.0, 0.0], [-1.409264, 2.440917, 0.0],
                [0.0, 0.0, 25.508255]],
    "coords": [[0.66667, 0.33333, 0.090221], [0.33333, 0.66667, 0.18043936],
               [0.0, 0.0, 0.27065772], [0.66667, 0.33333, 0.36087608],
               [0.33333, 0.66667, 0.45109444], [0.0, 0.0, 0.49656991]],
    "species": ['H', 'H', 'H', 'H', 'H', 'He'],
    "site_properties": {
        'SiteTypes': ['S1', 'S1', 'S1', 'S1', 'S1', 'A1']
    }
}
PRIMITIVE_CELL_INF0 = {
    "cutoff": np.around(6.00),
    "structure": PRIMITIVE_CELL,
    "aos": ['1', '0', '2'],
    "pbc": [True, True, False],
    "ns": 1,
    "na": 1
}


class _PtAdsorptionLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, 'Platinum_adsorption.json')
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=PLATINUM_URL,
                                             dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(
                os.path.join(self.data_dir, 'Platinum_adsorption.tar.gz'),
                self.data_dir)
        loader = dc.data.JsonLoader(tasks=PLATINUM_TASKS,
                                    feature_field="Structures",
                                    label_field="Formation Energy",
                                    featurizer=self.featurizer,
                                    **self.args)
        return loader.create_dataset(dataset_file)


def load_Platinum_Adsorption(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.SineCoulombMatrix(),
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
    are 648 diffrent adsorbate configuration in this datasets represented as Pymatgen
    Structure objects.

    1. Pymatgen structure object with site_properties with following key value.
        - "SiteTypes", mentioning if it is a active site "A1" or spectator
            site "S1".
        - "oss", diffrent occupational sites. For spectator sites make it -1.


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
    >> tasks, datasets, transformers = load_Platinum_Adsorption(
    >>    reload=True,
    >>    data_dir=data_path,
    >>    save_dir=data_path,
    >>    featurizer_kwargs=feat_args)
    >> train_dataset, val_dataset, test_dataset = datasets
    """

    loader = _PtAdsorptionLoader(featurizer, splitter, transformers,
                                 PLATINUM_TASKS, data_dir, save_dir, **kwargs)
    return loader.load_dataset('LCNN_feat', reload)
