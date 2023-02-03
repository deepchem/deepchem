"""
qm7 dataset loader.
"""
import os
import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator, _MolnetLoader
from deepchem.data import Dataset
from typing import List, Optional, Tuple, Union

QM7_MAT_UTL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat"
QM7_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv"
QM7B_MAT_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat"
GDB7_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb7.tar.gz"
QM7_TASKS = ["u0_atom"]


class _QM7Loader(_MolnetLoader):

    def create_dataset(self) -> Dataset:
        dataset_file = os.path.join(self.data_dir, "gdb7.sdf")
        if not os.path.exists(dataset_file):
            dc.utils.data_utils.download_url(url=GDB7_URL,
                                             dest_dir=self.data_dir)
            dc.utils.data_utils.untargz_file(
                os.path.join(self.data_dir, "gdb7.tar.gz"), self.data_dir)
        loader = dc.data.SDFLoader(tasks=self.tasks,
                                   featurizer=self.featurizer,
                                   sanitize=True)
        return loader.create_dataset(dataset_file, shard_size=8192)


def load_qm7(
    featurizer: Union[dc.feat.Featurizer, str] = dc.feat.CoulombMatrix(23),
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: List[Union[TransformerGenerator, str]] = ['normalization'],
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load QM7 dataset

    QM7 is a subset of GDB-13 (a database of nearly 1 billion
    stable and synthetically accessible organic molecules)
    containing up to 7 heavy atoms C, N, O, and S. The 3D
    Cartesian coordinates of the most stable conformations and
    their atomization energies were determined using ab-initio
    density functional theory (PBE0/tier2 basis set). This dataset
    also provided Coulomb matrices as calculated in [Rupp et al.
    PRL, 2012]:

    Stratified splitting is recommended for this dataset.

    The data file (.mat format, we recommend using `scipy.io.loadmat`
    for python users to load this original data) contains five arrays:

    - "X" - (7165 x 23 x 23), Coulomb matrices
    - "T" - (7165), atomization energies (unit: kcal/mol)
    - "P" - (5 x 1433), cross-validation splits as used in [Montavon et al.
        NIPS, 2012]
    - "Z" - (7165 x 23), atomic charges
    - "R" - (7165 x 23 x 3), cartesian coordinate (unit: Bohr) of each atom in
        the molecules

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

    Note
    ----
    DeepChem 2.4.0 has turned on sanitization for this dataset by
    default.  For the QM7 dataset, this means that calling this
    function will return 6838 compounds instead of 7160 in the source
    dataset file.  This appears to be due to valence specification
    mismatches in the dataset that weren't caught in earlier more lax
    versions of RDKit.  Note that this may subtly affect benchmarking
    results on this
    dataset.

    References
    ----------
    .. [1] Rupp, Matthias, et al. "Fast and accurate modeling of molecular
        atomization energies with machine learning." Physical review letters
        108.5 (2012): 058301.
    .. [2] Montavon, Grégoire, et al. "Learning invariant representations of
        molecules for atomization energy prediction." Advances in Neural
        Information Proccessing Systems. 2012.
    """
    loader = _QM7Loader(featurizer, splitter, transformers, QM7_TASKS, data_dir,
                        save_dir, **kwargs)
    return loader.load_dataset('qm7', reload)
