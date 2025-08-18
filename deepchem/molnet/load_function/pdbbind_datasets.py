"""
PDBBind dataset loader.
"""
import os
import numpy as np
import pandas as pd

import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union, Literal

DATASETS_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
PDBBIND_URL = DATASETS_URL + "pdbbindv2019/"
PDBBIND_TASKS = ['-logKd/Ki']

CollectionType = Literal['PL', 'PP', 'PN']
SetName = Literal['refined', 'general', 'core']

class _PDBBindLoader(_MolnetLoader):

    def __init__(self,
                 *args,
                 pocket: bool = True,
                 set_name: SetName = 'core',
                 collection: CollectionType = 'PL',
                 **kwargs):
        """Initialize PDBBind loader.
        
        Parameters
        ----------
        pocket: bool, optional (default True)
            If True, load only the binding pocket for protein-ligand complexes
        set_name: str, optional (default 'core')
            Dataset split to load. Options: 'refined', 'general', 'core'
        collection: str, optional (default 'PL')
            Collection type to load. Options:
            - 'PL': Protein-Ligand complexes
            - 'PP': Protein-Protein complexes
            - 'PN': Protein-Nucleic acid complexes
        """
        super(_PDBBindLoader, self).__init__(*args, **kwargs)
        self.pocket = pocket
        self.set_name = set_name
        self.collection = collection.upper()
        
        # Validate collection type
        if self.collection not in ['PL', 'PP', 'PN']:
            raise ValueError("collection must be one of: 'PL', 'PP', 'PN'")
        
        # Core set only available for PL collection
        if set_name == 'core' and collection != 'PL':
            raise ValueError("'core' set is only available for Protein-Ligand collection")
            
        # Map set names to folder names
        if set_name == 'general':
            self.name = f'pdbbind_v2019_other_{self.collection}'
        elif set_name == 'refined':
            self.name = f'pdbbind_v2019_{self.collection}_refined'
        elif set_name == 'core':
            self.name = 'pdbbind_v2013_core_set'

    def create_dataset(self) -> Dataset:
        """Create dataset from PDBBind data."""
        if self.set_name not in ['refined', 'general', 'core']:
            raise ValueError(
                "Only 'refined', 'general', and 'core' are supported for set_name."
            )

        filename = self.name + '.tar.gz'
        data_folder = os.path.join(self.data_dir, self.name)
        dataset_file = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(data_folder):
            if self.set_name in ['refined', 'general']:
                dc.utils.data_utils.download_url(url=PDBBIND_URL + filename,
                                              dest_dir=self.data_dir)
            else:
                dc.utils.data_utils.download_url(url=DATASETS_URL + filename,
                                              dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(dataset_file,
                                          dest_dir=self.data_dir)

        # get molecule files, labels and pdbids
        first_mol_files, second_mol_files, labels, pdbs = self._process_pdbs()

        # load and featurize each complex
        features = self.featurizer.featurize(
            list(zip(first_mol_files, second_mol_files)))
        dataset = dc.data.DiskDataset.from_numpy(features, y=labels, ids=pdbs)

        return dataset

    def _process_pdbs(
            self) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
        """Process PDB files based on collection type."""
        
        # Set up data folder and index file paths
        if self.set_name == 'general':
            data_folder = os.path.join(self.data_dir, f'v2019-other-{self.collection}')
            index_file = f'index/INDEX_general_{self.collection}_data.2019'
        elif self.set_name == 'refined':
            data_folder = os.path.join(self.data_dir, f'refined-set-{self.collection}')
            index_file = f'index/INDEX_refined_{self.collection}_data.2019'
        else:  # core set
            data_folder = os.path.join(self.data_dir, 'v2013-core')
            index_file = 'pdbbind_v2013_core.csv'
            
        index_labels_file = os.path.join(data_folder, index_file)

        if self.set_name in ['general', 'refined']:
            # Extract locations of data
            with open(index_labels_file, "r") as g:
                pdbs = [line[:4] for line in g.readlines() if line[0] != "#"]
            # Extract labels
            with open(index_labels_file, "r") as g:
                labels = np.array([float(line.split()[3])
                                 for line in g.readlines()
                                 if line[0] != "#"])
        else:
            df = pd.read_csv(index_labels_file)
            pdbs = df.pdb_id.tolist()
            labels = np.array(df.label.tolist())

        # Set up file paths based on collection type
        if self.collection == 'PL':
            if self.pocket:
                first_mol_files = [
                    os.path.join(data_folder, pdb, f"{pdb}_pocket.pdb")
                    for pdb in pdbs
                ]
            else:
                first_mol_files = [
                    os.path.join(data_folder, pdb, f"{pdb}_protein.pdb")
                    for pdb in pdbs
                ]
            second_mol_files = [
                os.path.join(data_folder, pdb, f"{pdb}_ligand.sdf")
                for pdb in pdbs
            ]
        else:
            # For PP and PN collections
            first_mol_files = [
                os.path.join(data_folder, pdb, f"{pdb}_protein1.pdb")
                for pdb in pdbs
            ]
            second_mol_files = [
                os.path.join(data_folder, pdb,
                            f"{pdb}_{'protein2' if self.collection == 'PP' else 'nucleic'}.pdb")
                for pdb in pdbs
            ]

        return (first_mol_files, second_mol_files, labels, pdbs)


def load_pdbbind(
    featurizer: dc.feat.ComplexFeaturizer,
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    pocket: bool = True,
    set_name: SetName = 'core',
    collection: CollectionType = 'PL',
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load PDBBind dataset.

    The PDBBind dataset includes experimental binding affinity data
    for three types of molecular complexes:
    1. Protein-Ligand (PL): 4852 complexes in refined set, 12800 in general set
    2. Protein-Protein (PP): Available in refined and general sets
    3. Protein-Nucleic acid (PN): Available in refined and general sets

    The refined set removes data with obvious problems in 3D structure,
    binding data, or other aspects. The general set contains additional
    data not included in the refined set. The core set (only available
    for protein-ligand complexes) is a subset of the refined set that
    is not updated annually.

    Parameters
    ----------
    featurizer: ComplexFeaturizer or str
        The complex featurizer to use for processing the data.
        Alternatively you can pass one of the names from
        dc.molnet.featurizers as a shortcut.
    splitter: Splitter or str
        The splitter to use for splitting the data into training, validation, and
        test sets. Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut. If this is None, all the data
        will be included in a single dataset.
    transformers: list of TransformerGenerators or strings
        The Transformers to apply to the data. Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.
    reload: bool
        If True, the first call for a particular featurizer and splitter will cache
        the datasets to disk, and subsequent calls will reload the cached datasets.
    data_dir: str
        A directory to save the raw data in
    save_dir: str
        A directory to save the dataset in
    pocket: bool (default True)
        If true, use only the binding pocket for protein-ligand complexes
    set_name: str (default 'core')
        Name of dataset to download. 'refined', 'general', and 'core' are supported.
    collection: str (default 'PL')
        Type of complexes to load:
        - 'PL': Protein-Ligand complexes
        - 'PP': Protein-Protein complexes
        - 'PN': Protein-Nucleic acid complexes
        Note: 'core' set is only available for 'PL' collection.

    Returns
    -------
    tasks, datasets, transformers: tuple
        tasks: list
            Column names corresponding to machine learning target variables.
        datasets: tuple
            train, validation, test splits of data as
            ``deepchem.data.datasets.Dataset`` instances.
        transformers: list
            ``deepchem.trans.transformers.Transformer`` instances applied
            to dataset.

    References
    ----------
    .. [1] Liu, Z.H. et al. Acc. Chem. Res. 2017, 50, 302-309. (PDBbind v.2016)
    .. [2] Liu, Z.H. et al. Bioinformatics, 2015, 31, 405-412. (PDBbind v.2014)
    .. [3] Li, Y. et al. J. Chem. Inf. Model., 2014, 54, 1700-1716.(PDBbind v.2013)
    .. [4] Cheng, T.J. et al. J. Chem. Inf. Model., 2009, 49, 1079-1093. (PDBbind v.2009)
    .. [5] Wang, R.X. et al. J. Med. Chem., 2005, 48, 4111-4119. (Original release)
    .. [6] Wang, R.X. et al. J. Med. Chem., 2004, 47, 2977-2980. (Original release)
    """
    loader = _PDBBindLoader(featurizer,
                           splitter,
                           transformers,
                           PDBBIND_TASKS,
                           data_dir,
                           save_dir,
                           pocket=pocket,
                           set_name=set_name,
                           collection=collection,
                           **kwargs)
    return loader.load_dataset(loader.name, reload)
