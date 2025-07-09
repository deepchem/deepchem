from deepchem.feat import Featurizer
import numpy as np
import pandas as pd
import logging
from typing import Any, Iterable
from deepchem.data import NumpyDataset
from typing import Tuple

logger = logging.getLogger(__name__)


class ACTINNFeaturizer(Featurizer):
    """
    This class is the implementation of all transformations performed to
    scRNA-seq (Single-cell RNA sequencing) data for cell type identification
    using the model ACTINN.

    Example:
    --------

    >>> import deepchem as dc
    >>> import pandas as pd
    >>> import os
    
    >>> dataset = os.path.join(os.path.dirname(__file__), "test",
                                "data","sc_rna_seq_data")
    >>> # train_set.shape = (cells x genes)
    >>> train_set = pd.read_hdf('test_data/train_set.h5')
    >>> Ntest_set = pd.read_hdf('test_data/test_set.h5')
    >>> train_set,test_set = featurizer.find_common_genes(train_set,test_set)
    >>> train_set.shape
    (1000,23015)

    >>> # cell types
    >>> train_labels = pd.read_csv('dataset/tma_both_cleaned_label.txt',sep='\t',header=None)
    >>> test_labels = pd.read_csv('dataset/tma_both_cleaned_label.txt',sep='\t',header=None)

    >>> featurizer = dc.feat.ACTINNFeaturizer()

    >>> train_dataset = featurizer.featurize(train_set,labels[1])
    >>> train_dataset
    <NumpyDataset X.shape: (56112, 22099), y.shape: (56112,), w.shape: (56112,), task_names: [0]>
    >>> test_dataset = featurizer.featurize(train_set,labels=labels[1],mode='test')
    >>> test_dataset
    <NumpyDataset X.shape: (56112, 23013), y.shape: (56112,), w.shape: (56112,), task_names: [0]>
    >>> test_dataset.X = featurizer.filter_genes(test_dataset)
    >>> test_dataset
    <NumpyDataset X.shape: (56112, 22099), y.shape: (56112,), w.shape: (56112,), task_names: [0]>

    References:
    -----------
    [1] https://academic.oup.com/bioinformatics/article/36/2/533/5540320

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
        train_df : pd.DataFrame
            genes X cells raw counts, genes as index, cells as columns

        Returns
        -------
        X_filtered : np.ndarray
            filtered matrix (genes_filtered X cells), float32

        """
        print('entered scale sets')
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

    def type2label_dict(self, types):
        """

        Turn types into labels

        Parameters:
        -----------
        types: List[str]
            Types of cell present in the data

        Returns:
        --------
        type_to_label_dict: Dict
            A python dictionary mapping cell type strings to
            integer labels

        """

        all_celltype = list(set(types))
        self.n_types = len(all_celltype)

        celltype_to_label_dict = {}

        for i in range(len(all_celltype)):
            celltype_to_label_dict[all_celltype[i]] = i
        return celltype_to_label_dict

    def convert_type2label(self, types, type_to_label_dict):
        """
        Convert types to labels

        Parameters:
        -----------
            types: List
                list of cell types
            type_to_label dictionary: Dict
                dictionary of cell types mapped to integer labels

        Returns:
        --------
            labels:
                list of integer labels

        """

        types = list(types)
        labels = list()
        for type in types:
            labels.append(type_to_label_dict[type])
        return labels

    def featurize(self,
                  data: Iterable[Any],
                  labels: Iterable[Any] = None,
                  mode: str = 'train',
                  **kwargs) -> 'NumpyDataset':
        """
        Override of the base class Featurizer's featurize function.

        Parameters
        ----------
        data : Iterable[Any]
            A pandas DataFrame or iterable containing raw scRNA-seq data.
        labels : Iterable[Any]
            Cell type labels corresponding to each data point.
        mode : str, default='train'
            Mode of operation: 'train' or 'test'.
        **kwargs :
            Additional arguments passed to the internal `_featurize` function.

        Returns
        -------
        NumpyDataset
            A dataset object containing the transformed features and corresponding labels.
        """
        self.labels = labels
        print('self.labels',self.labels)
        if mode not in ['train', 'test']:
            print("`mode` must be either 'train' or 'test'")
            return None

        try:
            features = self._featurize(data, mode, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to featurize data. Error: {e}")
            return None
        if self.labels is not None:
            type2label_dict = self.type2label_dict(labels)
            labels = self.convert_type2label(labels, type2label_dict)

        return NumpyDataset(features, labels)

    def _featurize(self, dataset: pd.DataFrame, mode, **kwargs) -> np.ndarray:
        """

            Parameters
            ----------
            train_set :
                A pandas dataframe containing the raw scRNA-seq data

            Returns
            -------
                np.ndarray :
                    A numpy array containing the scRNA-seq data after transformation

            """

        if mode == 'train':
            scaled_set = self.scale_sets(dataset)
            print('scale sets fineshed')
        else:
            scaled_set = self.normalise(dataset)
            mask = self.filter_genes()
            scaled_set = scaled_set[mask,:]

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
        dataset_df : pd.DataFrame
            genes X cells raw counts, genes as index, cells as columns

        Returns
        -------
        X : np.ndarray
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

    def filter_genes(self):
        """
        Returns a boolean mask to subset the test set and labels
        to include only the genes retained after filtering the training set.
        Parameters
        ----------
        data : np.ndarray
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
            print("Train set hasn't been processed yet.")
            return None

    def find_common_genes(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ 
        Retain only the common genes (rows) in both train and test datasets.

        Parameters
        ----------
        train_set : pd.DataFrame
            A pandas DataFrame (genes x cells) representing the training data, with gene IDs as the index.
        test_set : pd.DataFrame
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