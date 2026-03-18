"""
SIDER dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

SIDER_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
SIDER_TASKS = [
    'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
    'Product issues', 'Eye disorders', 'Investigations',
    'Musculoskeletal and connective tissue disorders',
    'Gastrointestinal disorders', 'Social circumstances',
    'Immune system disorders', 'Reproductive system and breast disorders',
    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
    'General disorders and administration site conditions',
    'Endocrine disorders', 'Surgical and medical procedures',
    'Vascular disorders', 'Blood and lymphatic system disorders',
    'Skin and subcutaneous tissue disorders',
    'Congenital, familial and genetic disorders', 'Infections and infestations',
    'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
    'Renal and urinary disorders',
    'Pregnancy, puerperium and perinatal conditions',
    'Ear and labyrinth disorders', 'Cardiac disorders',
    'Nervous system disorders', 'Injury, poisoning and procedural complications'
]


class _SiderLoader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "sider.csv.gz")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=SIDER_URL,
                                             dest_dir=self.data_dir)
        loader = dc.data.CSVLoader(tasks=self.tasks,
                                   feature_field="smiles",
                                   featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_sider(
    featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
    splitter: Union[dc.splits.Splitter, str, None] = 'scaffold',
    transformers: List[Union[TransformerGenerator, str]] = ['balancing'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load SIDER dataset

    The Side Effect Resource (SIDER) is a database of marketed
    drugs and adverse drug reactions (ADR). The version of the
    SIDER dataset in DeepChem has grouped drug side effects into
    27 system organ classes following MedDRA classifications
    measured for 1427 approved drugs.

    Random splitting is recommended for this dataset.

    The raw data csv file contains columns below:

    - "smiles": SMILES representation of the molecular structure
    - "Hepatobiliary disorders" ~ "Injury, poisoning and procedural
        complications": Recorded side effects for the drug. Please refer
        to http://sideeffects.embl.de/se/?page=98 for details on ADRs.

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
    .. [1] Kuhn, Michael, et al. "The SIDER database of drugs and side effects."
        Nucleic acids research 44.D1 (2015): D1075-D1079.
    .. [2] Altae-Tran, Han, et al. "Low data drug discovery with one-shot
        learning." ACS central science 3.4 (2017): 283-293.
    .. [3] Medical Dictionary for Regulatory Activities. http://www.meddra.org/
    """
    loader = _SiderLoader(featurizer, splitter, transformers, SIDER_TASKS,
                          data_dir, save_dir, **kwargs)
    return loader.load_dataset('sider', reload)
