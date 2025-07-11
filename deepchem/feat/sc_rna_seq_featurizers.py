from deepchem.feat import Featurizer

import numpy as np
import pandas as pd

import logging
from typing import Any, Iterable, Tuple, List

logger = logging.getLogger(__name__)


class ACTINNFeaturizer(Featurizer):
    """
    This class is the implementation of all transformations performed to
    scRNA-seq (Single-cell RNA sequencing) data for cell type identification
    using the model ACTINN.

    References
    -----------
    .. [1] https://academic.oup.com/bioinformatics/article/36/2/533/5540320

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.data import NumpyDataset
    >>> import pandas as pd
    >>> import os
    >>> featurizer = dc.feat.ACTINNFeaturizer()
    >>> dataset = os.path.join(os.path.dirname(__file__), "test",
    ...                            "data", "sc_rna_seq_data")
    >>> # train_set.shape = (genes x cells)
    >>> train_set = pd.read_hdf(os.path.join(dataset, "scRNAseq_sample_1.h5"))
    >>> train_set.shape
    (1500, 1000)
    >>> # test_set.shape = (genes x cells)
    >>> test_set = pd.read_hdf(os.path.join(dataset, "scRNAseq_sample_2.h5"))
    >>> test_set.shape
    (1200, 1000)
    >>> train_set,test_set = featurizer.find_common_genes(train_set, test_set)
    >>> train_set.shape
    (802, 1000)
    >>> test_set.shape
    (802, 1000)
    >>> # train_labels.shape = (1000,2). train_label.columns = (cell_id, cell_type)
    >>> train_labels = pd.read_csv('os.path.join(dataset, "train_label.txt")', sep='\t', header=None)
    >>> train_labels = train_labels.loc[:][1]
    >>> train_set = featurizer.featurize(train_set, labels[1], mode='train')
    >>> train_set.shape
    (1000, 768)
    >>> test_set = featurizer.featurize(train_set, labels=train_labels, mode='test')
    >>> test_set.shape
    (1000, 768)
    >>> train_dataset = dc.data.NumpyDataset(X=train_set,y=train_labels[1])
    >>> test_dataset = dc.data.NumpyDataset(X=test_set)
    """

    def scale_sets(self, dataset_df: pd.DataFrame) -> np.ndarray:
        """
        A function to perform data transformations to identify celltypes from scRNA-seq data
        using ACTINN. Normalised and filters out top and bottom 1% genes based on total expression
        and coefficient of variation across all cells.

        1) Convert once to float32 NumPy array
        2) Library-size normalize to 10,000 counts per cell
        3) Log2(x+1) transform
        4) Filter genes by total expression (1st–99th percentile)
        5) Filter genes by coefficient of variation (1st–99th percentile)

        Parameters
        ----------
        dataset_df: pd.DataFrame
            genes X cells raw counts, genes as index, cells as columns

        Returns
        -------
        X: np.ndarray
            filtered matrix (genes_filtered X cells), float32
        """

        # 1) extract gene names & data array
        gene_names = dataset_df.index.to_numpy()
        X = np.array(dataset_df, dtype=np.float32)

        # 2) library-size normalize to 10,000 (in-place)
        col_sums = X.sum(axis=0, keepdims=True)  # shape (1, n_cells)
        X /= col_sums  # broadcast divide
        X *= 10000

        # 3) log2(x + 1) transform (in-place)
        np.log2(X + 1, out=X)

        # 4) filter by total expression
        expr = X.sum(axis=1)  # total per gene
        low, high = np.percentile(expr, [1, 99])
        mask_expr = (expr >= low) & (expr <= high)
        X = X[mask_expr, :]
        gene_names = gene_names[mask_expr]

        # remove genes with zero mean
        mean_expr = X.mean(axis=1)
        mask_mean = mean_expr > 0
        X = X[mask_mean, :]
        gene_names = gene_names[mask_mean]

        # 5) filter by coefficient of variation
        mean_expr = X.mean(axis=1)
        cv = X.std(axis=1) / mean_expr
        low_cv, high_cv = np.percentile(cv, [1, 99])
        mask_cv = (cv >= low_cv) & (cv <= high_cv)
        X = X[mask_cv, :]
        gene_names = gene_names[mask_cv]

        self.gene_list = gene_names

        return X

    def convert_type2label(self, types: Iterable[Any]) -> List[int]:
        """
        Convert cell types to integer labels

        Parameters
        ----------
        types: Iterable
            list of cell types

        Returns
        -------
        labels:
            list of integer labels
        """
        all_celltype = list(set(types))
        self.n_types = len(all_celltype)

        type_to_label_dict = {}

        for i in range(len(all_celltype)):
            type_to_label_dict[all_celltype[i]] = i

        types = list(types)
        labels = list()
        for type in types:
            labels.append(type_to_label_dict[type])
        return labels

    # def featurize(self,
    #               datapoints: Iterable[Any],
    #               log_every_n: int = 1000,
    #               **kwargs) -> np.ndarray:
    #     """
    #     Converts raw scRNA-seq data into model-ready features and labels.
    #     It handles preprocessing based on the specified mode (`train` or `test`)
    #     and transforms raw data into a deepchem `NumpyDataset` containing feature
    #     arrays and optional label arrays.

    #     Parameters
    #     ----------
    #     datapoints: pd.Dataframe
    #         A pandas DataFrame or iterable containing raw scRNA-seq data.

    #     Returns
    #     -------
    #     NumpyDataset
    #         A dataset object containing the transformed features and corresponding labels.
    #     """
    #     try:
    #         features = self._featurize(datapoints, **kwargs)
    #     except Exception as e:
    #         logger.warning(f"Failed to featurize data. Error: {e}")
    #         return np.array([])

    #     return features
    def featurize(self,
                  datapoints: Iterable[Any],
                  log_every_n: int = 1000,
                  **kwargs) -> np.ndarray:
        """  Converts raw scRNA-seq data into model-ready features and labels.

            Parameters
            ----------
            datapoints: Iterable[Any]
                Pandas Dataframe containing the dataset

            Returns
            -------
            np.ndarray
                A numpy array containing a featurized representation of `datapoints`.
        """
        try:
            features = self._featurize(datapoints, **kwargs)
        except:
            logger.warning(
                "Failed to featurize datapoint %d. Appending empty array")
            features = np.array([])
        return np.asarray(features)

    def _featurize(self, datapoint: Any, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        datapoint:
            A pandas dataframe containing the raw scRNA-seq data

        Returns
        -------
        np.ndarray:
            A numpy array containing the scRNA-seq data after transformation
        """
        scaled_set = self.scale_sets(datapoint)

        # gens x cells --> cells x gens
        dataset = np.transpose(scaled_set)
        return dataset

    def featurize_testset(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Function to featurize test set after train set featurisation
        """
        scaled_set = self.normalise(dataset)
        mask = self.filter_genes()
        scaled_set = scaled_set[mask, :]

        # gens x cells --> cells x gens
        dataset = np.transpose(scaled_set)

        return dataset

    def normalise(self, dataset_df: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Normalise scRNA-seq data
        1) Library-size normalize to 10,000 counts per cell
        2) Log2(x+1) transform

        Parameters
        ----------
        dataset_df: pd.DataFrame
            genes X cells raw counts, genes as index, cells as columns

        Returns
        -------
        X: np.ndarray
            normalised matrix (genes X cells), float32
        """

        # 1) extract gene names & data array
        self.test_genes = dataset_df.index.to_numpy()
        X = np.array(dataset_df, dtype=np.float32)

        # 2) library-size normalize to 10,000 (in-place)
        col_sums = X.sum(axis=0, keepdims=True)  # shape (1, n_cells)
        X /= col_sums  # broadcast divide
        X *= 10000

        # 3) log2(x + 1) transform (in-place)
        np.log2(X + 1, out=X)

        return X

    def filter_genes(self) -> np.ndarray:
        """
        Returns a boolean mask to subset the test set and labels
        to include only the genes retained after filtering the training set.

        Parameters
        ----------
        data: np.ndarray
            A normalized matrix of shape (cells x genes).

        Returns
        -------
        np.ndarray
            A subset of the input DataFrame containing only the filtered genes.
        """
        if hasattr(self, "gene_list") and self.gene_list.size > 0:
            mask = np.isin(self.test_genes, self.gene_list)
            return mask
        else:
            return np.zeros_like(self.test_genes, dtype=bool)

    def find_common_genes(
            self, train_set: pd.DataFrame,
            test_set: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retain only the common genes (rows) in both train and test datasets.

        Parameters
        ----------
        train_set: pd.DataFrame
            A pandas DataFrame (genes x cells) representing the training data, with gene IDs as the index.
        test_set: pd.DataFrame
            A pandas DataFrame (genes x cells) representing the testing data, with gene IDs as the index.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Filtered train and test DataFrames containing only the common genes.
        """
        common_genes = train_set.index.intersection(test_set.index)
        common_genes = sorted(common_genes)

        train_set = train_set.loc[common_genes]
        test_set = test_set.loc[common_genes]

        return train_set, test_set
