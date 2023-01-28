import logging
import numpy as np
from typing import Tuple
from deepchem.data import Dataset
from deepchem.trans.transformers import Transformer

logger = logging.getLogger(__name__)


class DuplicateBalancingTransformer(Transformer):
    """Balance binary or multiclass datasets by duplicating rarer class samples.

    This class balances a dataset by duplicating samples of the rarer class so
    that the sum of all example weights from all classes is the same. (Up to
    integer rounding of course). This can be useful when you're working on an
    imabalanced dataset where there are far fewer examples of some classes than
    others.

    This class differs from `BalancingTransformer` in that it actually
    duplicates rarer class samples rather than just increasing their sample
    weights. This may be more friendly for models that are numerically fragile
    and can't handle imbalanced example weights.

    Examples
    --------
    Here's an example for a binary dataset.

    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> n_classes = 2
    >>> import deepchem as dc
    >>> import numpy as np
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    >>> dataset = transformer.transform(dataset)

    And here's a multiclass dataset example.

    >>> n_samples = 50
    >>> n_features = 3
    >>> n_tasks = 1
    >>> n_classes = 5
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.DuplicateBalancingTransformer(dataset=dataset)
    >>> dataset = transformer.transform(dataset)

    See Also
    --------
    deepchem.trans.BalancingTransformer: Balance by changing sample weights.

    Note
    ----
    This transformer is only well-defined for singletask datasets. (Since
    examples are actually duplicated, there's no meaningful way to duplicate
    across multiple tasks in a way that preserves the balance.)

    This transformer is only meaningful for classification datasets where `y`
    takes on a limited set of values. This class transforms all of `X`, `y`,
    `w`, `ids`.

    Raises
    ------
    `ValueError` if the provided dataset is multitask.
    """

    def __init__(self, dataset: Dataset):
        super(DuplicateBalancingTransformer, self).__init__(transform_X=True,
                                                            transform_y=True,
                                                            transform_w=True,
                                                            transform_ids=True,
                                                            dataset=dataset)

        if len(dataset.get_task_names()) > 1:
            raise ValueError(
                "This transformation is only defined for singletask datsets.")

        # Get the labels/weights
        y = dataset.y
        w = dataset.w
        # Normalize shapes
        if len(y.shape) == 1:
            y = np.reshape(y, (len(y), 1))
        if len(w.shape) == 1:
            w = np.reshape(w, (len(w), 1))
        if len(y.shape) != 2:
            raise ValueError("y must be of shape (N,) or (N, n_tasks)")
        if len(w.shape) != 2:
            raise ValueError("w must be of shape (N,) or (N, n_tasks)")
        self.classes = sorted(np.unique(y))
        # Remove labels with zero weights
        y = y[w != 0]
        class_weights = []
        # Note that we may have 0 elements of a given class since we remove those
        # labels with zero weight.
        for c in self.classes:
            # this works because y is 1D
            c_weight = np.sum(w[y == c])
            class_weights.append(c_weight)
        weight_largest = max(class_weights)
        # This is the right ratio since int(N/num_c) * num_c \approx N
        # for all classes
        duplication_ratio = [
            int(weight_largest / float(c_weight)) if c_weight > 0 else 0
            for c_weight in class_weights
        ]
        self.duplication_ratio = duplication_ratio

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, id) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idtrans: np.ndarray
            Transformed array of identifiers
        """
        if not (len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)):
            raise ValueError("y must be of shape (N,) or (N, 1)")
        if not (len(w.shape) == 1 or (len(w.shape) == 2 and w.shape[1] == 1)):
            raise ValueError("w must be of shape (N,) or (N, 1)")
        # Flattening is safe because of shape check above
        y = y.flatten()
        w = w.flatten()
        X_dups, y_dups, w_dups, ids_dups = [], [], [], []
        for i, c in enumerate(self.classes):
            duplication_ratio = self.duplication_ratio[i]
            c_inds = (y == c)
            X_c = X[c_inds]
            y_c = y[c_inds]
            w_c = w[c_inds]
            ids_c = ids[c_inds]
            X_c_dup = np.repeat(X_c, duplication_ratio, axis=0)
            y_c_dup = np.repeat(y_c, duplication_ratio, axis=0)
            w_c_dup = np.repeat(w_c, duplication_ratio, axis=0)
            ids_c_dup = np.repeat(ids_c, duplication_ratio, axis=0)
            X_dups.append(X_c_dup)
            y_dups.append(y_c_dup)
            w_dups.append(w_c_dup)
            ids_dups.append(ids_c_dup)
        Xtrans = np.concatenate(X_dups, axis=0)
        ytrans = np.concatenate(y_dups, axis=0)
        wtrans = np.concatenate(w_dups, axis=0)
        idstrans = np.concatenate(ids_dups, axis=0)
        return (Xtrans, ytrans, wtrans, idstrans)
