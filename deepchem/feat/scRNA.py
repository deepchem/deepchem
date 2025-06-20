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
    >>> train_set = pd.read_csv(train_set.csv)
    >>> train_set.shape
    (1000,23015)
    
    >>> featurizer = dc.feat.ACTINNFeaturizer()
    >>> features = featurizer.featurize(train_set)
    >>> features.shape
    (1000,18469)
    
    >>> loader = dc.data.CSVLoader(tasks=['label'], feature_field = list(train_set.columns)[1:], featurizer=dc.feat.ACTINNFeaturizer())
    >>> dataset = loader.create_dataset('dataset/train_set.csv')
    >>> print(dataset)
    <DiskDataset X.shape: (np.int64(1000), np.int64(18469)), y.shape: (np.int64(1000), np.int64(1)), w.shape: (np.int64(1000), np.int64(1)), task_names: ['label']>
    
    References:
    -----------
    [1] https://academic.oup.com/bioinformatics/article/36/2/533/5540320

    """

    def scale_sets(self, sets) -> List:
        
        """
        
        A function to perform data transformations to identify celltypes from scRNA-seq data
        using ACTINN. Takes in both train and test set and retains only the common genes in 
        both. Filters out top and bottom 1% genes based on total expression and coefficient of 
        variation across all cells (both train and test set). 
        
        Parameters:
        -----------
            sets:
                A list containing train and test sets in the shape (genes x cells)
        
        Returns:
        --------
            List:
                A list containing train and test sets after transformation is performed
        
        """
        
        # Retaining only genes that are common in both train and test set
        common_genes = set(sets[0].index)
        for i in range(1, len(sets)):
            common_genes = set.intersection(set(sets[i].index),common_genes)
        common_genes = sorted(list(common_genes))
        sep_point = [0]
        for i in range(len(sets)):
            sets[i] = sets[i].loc[common_genes,]
            sep_point.append(sets[i].shape[1])
        
        total_set_df = pd.concat(sets, axis=1, sort=False)
        total_set = np.array(total_set_df, dtype=np.float32)

        # To avoid getting NaNs when normalising across cells due to 0 mean
        col_sums = np.sum(total_set, axis=0, keepdims=True)
        nonzero_mask = col_sums != 0
        normalized = np.zeros_like(total_set)
        normalized[:, nonzero_mask[0]] = (
            total_set[:, nonzero_mask[0]] / col_sums[:, nonzero_mask[0]] * 20000
        )
        total_set = np.log2(normalized + 1)

        # Filtering out top and bottom 1 % genes(rows) based on total expression
        expr = np.sum(total_set, axis=1)
        total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]

        
        # Filter out rows(genes) with zero mean before calculating CV (coeffient of variation = std/mean)
        mean_expr = np.mean(total_set, axis=1)
        non_zero_mean_mask = mean_expr > 0
        total_set = total_set[non_zero_mean_mask, :]
        mean_expr = mean_expr[non_zero_mean_mask]
        
        # Filtering out top and bottom 1 % genes(rows) based on CV
        cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
        total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
        for i in range(len(sets)):
            sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
        return sets
    
    
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
    
    
    def _featurize(self, train_set, **kwargs) -> np.ndarray:
            """
            Parameters
            ----------
            train_set :
                A pandas dataframe containing the raw scRNA-seq data
                
            Returns
            -------
                np.ndarray:
                    A numpy array containing the scRNA-seq data after transformation 
            
            """
            # cells x genes -> genes x cells
            train_set = np.transpose(train_set)
            
            scaled_sets = self.scale_sets([train_set])
            train_set = scaled_sets[0]
        
            # gens x cells --> cells x gens
            train_set = np.transpose(train_set)
            
            return train_set
            