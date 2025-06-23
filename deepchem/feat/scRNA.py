from deepchem.feat import Featurizer
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from typing import Any, Iterable, List
import numpy as np



class ACTINNFeaturizer(Featurizer):
    """
    This class is the implementation of all transformations performed to 
    scRNA-seq (Single-cell RNA sequencing) data for cell type identification using the model
    ACTINN. 
    
    Example:
    --------
    
    >>> import deepchem as dc
    >>> import pandas as pd
    
    >>> # train_set.shape = (cells x genes) and train_set['label'] = cell type label 
    >>> train_set = pd.read_csv("train_set.csv")
    >>> train_set.shape
    (1000,23015)
    
    >>> featurizer = dc.feat.ACTINNFeaturizer()
    
    >>> train_loader = dc.data.CSVLoader(tasks=['label'], feature_field = list(train_set.columns)[1:], featurizer=featurizer)
    >>> train_dataset = train_loader.create_dataset('dataset/train_set.csv')
    >>> print(train_dataset)
    <DiskDataset X.shape: (np.int64(1000), np.int64(18469)), y.shape: (np.int64(1000), np.int64(1)), w.shape: (np.int64(1000), np.int64(1)), task_names: ['label']>
    
    >>> test_set = pd.read_csv("test_set.csv")
    >>> test_set.shape
    (1000,23015)
    >>> test_set_filtered = test_set[featurizer.gene_list]
    (1000, 18490)
    
    # to do normalisation without gene filtering as it is already done
    >>> featurizer_test = dc.feat.ACTINNFeaturizer(mode='test') 

    >>> test_loader = dc.data.CSVLoader(tasks=['label'], feature_field = test_set_filtered.columns featurizer=featurizer)
    >>> test_dataset = test_loader.create_dataset('dataset/test_set.csv')
    >>> print(test_dataset)
    <DiskDataset X.shape: (np.int64(1000), np.int64(18490)), y.shape: (np.int64(1000), np.int64(1)), w.shape: (np.int64(1000), np.int64(1)), task_names: ['label']>
    
    
    References:
    -----------
    [1] https://academic.oup.com/bioinformatics/article/36/2/533/5540320

    """
    
    def __init__(self, mode : str = 'train'):
        self.mode = mode   
        
    def scale_sets(self, dataset_df : pd.DataFrame) -> np.ndarray:
        
        """
        
        A function to perform data transformations to identify celltypes from scRNA-seq data
        using ACTINN. Normalised and filters out top and bottom 1% genes based on total expression a
        nd coefficient of variation across all cells. 
            
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
        
        # 1) extract gene names & data array
        gene_names = dataset_df.index.to_numpy()
        X = dataset_df.values.astype(np.float32, copy=True)

        # 2) library-size normalize to 10,000 (in-place)
        col_sums = X.sum(axis=0, keepdims=True)     # shape (1, n_cells)
        X /= col_sums                               # broadcast divide
        X *= 10000

        # 3) log2(x + 1) transform (in-place)
        np.log2(X + 1, out=X)

        # 4) filter by total expression
        expr = X.sum(axis=1)                        # total per gene
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

    def featurize(self,
                  datapoint: Iterable[Any],
                  **kwargs) -> np.ndarray:
        """
        To override the implementation of featurize function under the base 
        class (Class Featurizer)
        
        Parameters
        ----------
        datapoint : 
            A pandas dataframe containing the raw scRNA-seq data

        Returns
        -------
            np.ndarray:
                A numpy array containing the scRNA-seq data after transformation 
        """
        try:
            features = self._featurize(datapoint, **kwargs)
        except:
            logger.warning(
                "Failed to featurize datapoint %d. Appending empty array")
            features.append(np.array([]))

        return features
    
    
    def _featurize(self, dataset: pd.DataFrame, **kwargs) -> np.ndarray:
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
            # cells x genes -> genes x cells
            dataset = np.transpose(dataset)
            
            if self.mode == 'train':
                scaled_set = self.scale_sets(dataset)
            else:
                scaled_set = self.normalise(dataset)
                
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
        
        X = dataset_df.values.astype(np.float32, copy=True)

        # 1) library-size normalize to 10,000 (in-place)
        col_sums = X.sum(axis=0, keepdims=True)     # shape (1, n_cells)
        X /= col_sums                               # broadcast divide
        X *= 10000

        # 2) log2(x + 1) transform (in-place)
        np.log2(X + 1, out=X)
        
        return X