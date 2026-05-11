"""Representative protein structure loader for MoleculeNet.

This module provides a MoleculeNet-compatible loader for a small,
CATH-inspired set of representative protein structures. It downloads
PDB files from RCSB and featurizes backbone coordinates for use with
DeepChem models such as RFDiffusionModel.

The loader does not download the full CATH S40 dataset and does not
provide CATH hierarchy labels. The label array contains a single zero
placeholder task for API compatibility with supervised DeepChem datasets.

References
----------
.. [1] Sillitoe, I., et al. "CATH: expanded annotation and classification
   of protein structure." Nucleic Acids Research 49.D1 (2021): D266-D273.
"""
import os
import logging
import time
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import (TransformerGenerator,
                                                         _MolnetLoader)
from deepchem.data import Dataset, DiskDataset

logger = logging.getLogger(__name__)

CATH_TASKS = ['structure_placeholder']


class _CATHLoader(_MolnetLoader):
    """Loader for a representative protein structure subset.

    The default PDB ID list is inspired by the diversity of CATH folds,
    but it is a small RCSB PDB subset for prototyping and tests. It is
    not the complete CATH S40 dataset.
    """

    def __init__(self,
                 *args,
                 max_length: int = 512,
                 pdb_ids: Optional[Sequence[str]] = None,
                 max_download_attempts: int = 3,
                 download_timeout: float = 30.0,
                 **kwargs):
        """Initialize CATH loader.

        Parameters
        ----------
        max_length : int, default 512
            Maximum protein length to load. Longer proteins are truncated.
        pdb_ids : sequence of str, optional
            PDB IDs to load. If None, a small built-in representative set is
            used.
        max_download_attempts : int, default 3
            Maximum number of download attempts per missing PDB file.
        download_timeout : float, default 30.0
            Timeout in seconds for each RCSB download request.
        """
        super(_CATHLoader, self).__init__(*args, **kwargs)
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if max_download_attempts <= 0:
            raise ValueError("max_download_attempts must be positive")
        if download_timeout <= 0:
            raise ValueError("download_timeout must be positive")
        if pdb_ids is not None and len(pdb_ids) == 0:
            raise ValueError("pdb_ids must contain at least one PDB ID")
        self.name = 'cath_representative_pdb'
        self.max_length = max_length
        self.pdb_ids = list(pdb_ids) if pdb_ids is not None else None
        self.max_download_attempts = max_download_attempts
        self.download_timeout = download_timeout

    def create_dataset(self) -> Dataset:
        """Create the CATH dataset.

        Returns
        -------
        Dataset
            A DeepChem Dataset object containing protein structures.
        """
        # Get list of PDB IDs for a representative set
        pdb_ids = self._get_cath_pdb_list()

        # Download PDB files
        pdb_files, labels, ids = self._download_pdbs(pdb_ids)

        # Featurize proteins
        if len(pdb_files) == 0:
            raise ValueError("No PDB files available for featurization")

        features = self.featurizer.featurize(pdb_files)

        # Filter out failed featurizations (empty arrays)
        # Features is a list of arrays with variable lengths
        valid_data = [
            (f, labels[i], ids[i]) for i, f in enumerate(features) if f.size > 0
        ]

        if len(valid_data) == 0:
            raise ValueError("All featurizations failed")

        features = [d[0] for d in valid_data]
        labels = np.array([d[1] for d in valid_data])
        ids = [d[2] for d in valid_data]

        # Convert to object array to handle variable-length proteins
        features_array = np.empty(len(features), dtype=object)
        for i, f in enumerate(features):
            features_array[i] = f

        return DiskDataset.from_numpy(X=features_array,
                                      y=labels,
                                      ids=ids,
                                      tasks=self.tasks)

    def _get_cath_pdb_list(self) -> List[str]:
        """Get list of representative PDB IDs.

        Returns a representative set of diverse protein structures for
        testing and prototyping, or the user-provided ``pdb_ids``.

        Returns
        -------
        List[str]
            List of PDB IDs.
        """
        if self.pdb_ids is not None:
            return list(self.pdb_ids)

        # Representative set covering a variety of structural classes.
        representative_set = [
            "1CRN",  # Crambin - small
            "1UBQ",  # Ubiquitin - beta-grasp
            "2IG2",  # Immunoglobulin
            "1YCR",  # Rubredoxin
            "1A3N",  # Beta-propeller
            "1BKR",  # Alpha-beta plait
            "1TEN",  # Tenascin
            "1VII",  # Viral protein
            "1L2Y",  # Cytochrome
            "1E0L",  # Enolase
            "1FKB",  # FK506 binding
            "1GAB",  # G protein
            "1SRL",  # Serpentine receptor
            "1PHT",  # Phosphotransferase
            "1MBN",  # Myoglobin
            "3ICB",  # Immunoglobulin
            "256B",  # Cytochrome b
            "1HRC",  # Heme protein
        ]
        return representative_set

    def _download_pdbs(
            self,
            pdb_ids: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
        """Download PDB files from RCSB.

        Downloads each PDB file individually from the RCSB Protein Data
        Bank. Files are cached locally so subsequent calls skip the
        download step.

        Parameters
        ----------
        pdb_ids : List[str]
            List of PDB IDs to download.

        Returns
        -------
        Tuple[List[str], np.ndarray, List[str]]
            Tuple of (pdb_files, labels, ids) where:
            - pdb_files: List of paths to downloaded PDB files
            - labels: zero placeholders for API compatibility
            - ids: List of PDB IDs for successfully downloaded files
        """
        data_folder = os.path.join(self.data_dir, self.name)
        os.makedirs(data_folder, exist_ok=True)

        pdb_files = []
        ids = []
        missing_ids = []

        for pdb_id in pdb_ids:
            pdb_code = pdb_id.lower()
            pdb_file = os.path.join(data_folder, f"{pdb_code}.pdb")

            # Download if not cached
            if not os.path.exists(pdb_file):
                url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
                for attempt in range(self.max_download_attempts):
                    try:
                        with urlopen(url,
                                     timeout=self.download_timeout) as response:
                            status = getattr(response, 'status',
                                             response.getcode())
                            if status != 200:
                                logger.warning(
                                    "Failed to download %s (HTTP %d)", pdb_id,
                                    status)
                                if 400 <= status < 500:
                                    break
                                if attempt < self.max_download_attempts - 1:
                                    time.sleep(2**attempt)
                                continue
                            with open(pdb_file, 'wb') as f:
                                f.write(response.read())
                            break
                    except HTTPError as e:
                        logger.warning("Failed to download %s (HTTP %d)",
                                       pdb_id, e.code)
                        if 400 <= e.code < 500:
                            break
                        if attempt < self.max_download_attempts - 1:
                            time.sleep(2**attempt)
                    except (URLError, TimeoutError, OSError) as e:
                        logger.warning(
                            "Download attempt %d/%d for %s failed: %s",
                            attempt + 1, self.max_download_attempts, pdb_id, e)
                        if attempt < self.max_download_attempts - 1:
                            time.sleep(2**attempt)
                    except Exception as e:
                        logger.warning("Unexpected error downloading %s: %s",
                                       pdb_id, e)
                        break

            if os.path.exists(pdb_file):
                pdb_files.append(pdb_file)
                ids.append(pdb_id)
            else:
                missing_ids.append(pdb_id)

        if missing_ids:
            raise ValueError("Failed to download PDB IDs: %s" %
                             ", ".join(missing_ids))

        # Create zero placeholder labels for API compatibility.
        labels = np.zeros((len(ids), 1), dtype=np.float32)

        return pdb_files, labels, ids


def load_cath(
    featurizer: Union[dc.feat.Featurizer, str] = 'ProteinBackbone',
    splitter: Union[dc.splits.Splitter, str, None] = 'random',
    transformers: Optional[List[Union[TransformerGenerator, str]]] = None,
    reload: bool = True,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    max_length: int = 512,
    pdb_ids: Optional[Sequence[str]] = None,
    max_download_attempts: int = 3,
    download_timeout: float = 30.0,
    **kwargs
) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
    """Load a representative protein structure dataset.

    This loader provides access to a small CATH-inspired representative set
    of RCSB PDB structures suitable for prototyping protein structure
    generation models. It does not download the full CATH S40 dataset and
    does not provide CATH class labels.

    This dataset is particularly useful for:

    - Training protein backbone diffusion models (e.g., RFDiffusion)
    - Protein structure prediction tasks
    - Learning structural representations of proteins
    - Benchmarking structure-based models

    The dataset contains protein backbone coordinates (N, CA, C atoms)
    extracted from PDB structures.

    Random splitting is recommended for this dataset.

    Parameters
    ----------
    featurizer : Featurizer or str, default 'ProteinBackbone'
        The featurizer to use for processing protein structures.
        Use 'ProteinBackbone' for backbone coordinates.
        Alternatively, pass a custom Featurizer instance.
    splitter : Splitter or str, optional
        The splitter to use for splitting the data into training, validation,
        and test sets. Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut. If this is None, all the data
        will be included in a single dataset.
    transformers : list of TransformerGenerators or strings, optional
        The Transformers to apply to the data. Each one is specified by a
        TransformerGenerator or, as a shortcut, one of the names from
        dc.molnet.transformers.
    reload : bool, default True
        If True, the first call for a particular featurizer and splitter will
        cache the datasets to disk, and subsequent calls will reload the
        cached datasets.
    data_dir : str, optional
        A directory to save the raw data in.
    save_dir : str, optional
        A directory to save the dataset in.
    max_length : int, default 512
        Maximum protein length. Longer proteins will be truncated.
    pdb_ids : sequence of str, optional
        PDB IDs to load. If None, a small built-in representative set is used.
    max_download_attempts : int, default 3
        Maximum number of download attempts per missing PDB file.
    download_timeout : float, default 30.0
        Timeout in seconds for each RCSB download request.

    Returns
    -------
    tasks : list
        Column names corresponding to machine learning target variables.
        This loader returns one zero placeholder task for API compatibility.
    datasets : tuple
        Train, validation, test splits of data as
        ``deepchem.data.Dataset`` instances.
    transformers : list
        ``deepchem.trans.Transformer`` instances applied to dataset.

    Examples
    --------
    >>> import deepchem as dc
    >>> tasks, datasets, transformers = dc.molnet.load_cath(
    ...     featurizer='ProteinBackbone',
    ...     splitter='random'
    ... )
    >>> train, valid, test = datasets
    >>> print(train.X.shape)  # doctest: +SKIP

    References
    ----------
    .. [1] Sillitoe, I., et al. "CATH: expanded annotation and classification
       of protein structure." Nucleic Acids Research 49.D1 (2021): D266-D273.
    .. [2] Dawson, N. L., et al. "CATH: an expanded resource to predict protein
       function through structure and sequence." Nucleic Acids Research 45.D1
       (2017): D289-D295.
    .. [3] Watson, J. L., et al. "De novo design of protein structure and function
       with RFdiffusion." Nature 620.7976 (2023): 1089-1100.

    Notes
    -----
    This loader downloads PDB files from the RCSB Protein Data Bank.
    An internet connection is required on first use unless all requested PDB
    files already exist in the cache directory.
    """
    # Handle featurizer string shortcut
    if isinstance(featurizer, str) and featurizer.lower() == 'proteinbackbone':
        featurizer = dc.feat.ProteinBackboneFeaturizer(max_length=max_length)
    if transformers is None:
        transformers = []

    loader = _CATHLoader(featurizer,
                         splitter,
                         transformers,
                         CATH_TASKS,
                         data_dir,
                         save_dir,
                         max_length=max_length,
                         pdb_ids=pdb_ids,
                         max_download_attempts=max_download_attempts,
                         download_timeout=download_timeout,
                         **kwargs)
    return loader.load_dataset(loader.name, reload)
