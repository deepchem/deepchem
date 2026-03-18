"""
Contains an abstract base class that supports data transformations.
"""
import os
import logging
import time
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.ndimage

import deepchem as dc
from deepchem.data import Dataset, NumpyDataset, DiskDataset
from deepchem.feat import Featurizer
from deepchem.feat.mol_graphs import ConvMol

logger = logging.getLogger(__name__)


def undo_grad_transforms(grad, tasks, transformers):
    """DEPRECATED. DO NOT USE."""
    logger.warning(
        "undo_grad_transforms is DEPRECATED and will be removed in a future version of DeepChem. "
        "Manually implement transforms to perform force calculations.")
    for transformer in reversed(transformers):
        if transformer.transform_y:
            grad = transformer.untransform_grad(grad, tasks)
    return grad


def get_grad_statistics(dataset):
    """Computes and returns statistics of a dataset

    DEPRECATED DO NOT USE.

    This function assumes that the first task of a dataset holds the
    energy for an input system, and that the remaining tasks holds the
    gradient for the system.
    """
    logger.warning(
        "get_grad_statistics is DEPRECATED and will be removed in a future version of DeepChem. Manually compute force/energy statistics."
    )
    if len(dataset) == 0:
        return None, None, None, None
    y = dataset.y
    energy = y[:, 0]
    grad = y[:, 1:]
    for i in range(energy.size):
        grad[i] *= energy[i]
    ydely_means = np.sum(grad, axis=0) / len(energy)
    return grad, ydely_means


class Transformer(object):
    """Abstract base class for different data transformation techniques.

    A transformer is an object that applies a transformation to a given
    dataset. Think of a transformation as a mathematical operation which
    makes the source dataset more amenable to learning. For example, one
    transformer could normalize the features for a dataset (ensuring
    they have zero mean and unit standard deviation). Another
    transformer could for example threshold values in a dataset so that
    values outside a given range are truncated. Yet another transformer
    could act as a data augmentation routine, generating multiple
    different images from each source datapoint (a transformation need
    not necessarily be one to one).

    Transformers are designed to be chained, since data pipelines often
    chain multiple different transformations to a dataset. Transformers
    are also designed to be scalable and can be applied to
    large `dc.data.Dataset` objects. Not that Transformers are not
    usually thread-safe so you will have to be careful in processing
    very large datasets.

    This class is an abstract superclass that isn't meant to be directly
    instantiated. Instead, you will want to instantiate one of the
    subclasses of this class inorder to perform concrete
    transformations.
    """
    # Hack to allow for easy unpickling:
    # http://stefaanlippens.net/pickleproblem
    __module__ = os.path.splitext(os.path.basename(__file__))[0]

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 transform_w: bool = False,
                 transform_ids: bool = False,
                 dataset: Optional[Dataset] = None):
        """Initializes transformation based on dataset statistics.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        transform_w: bool, optional (default False)
            Whether to transform w
        transform_ids: bool, optional (default False)
            Whether to transform ids
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        """
        if self.__class__.__name__ == "Transformer":
            raise ValueError(
                "Transformer is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
            )
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.transform_w = transform_w
        self.transform_ids = transform_ids
        # Some transformation must happen
        assert transform_X or transform_y or transform_w or transform_ids

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        raise NotImplementedError(
            "Each Transformer is responsible for its own transform_array method."
        )

    def untransform(self, transformed: np.ndarray) -> np.ndarray:
        """Reverses stored transformation on provided data.

        Depending on whether `transform_X` or `transform_y` or `transform_w` was
        set, this will perform different un-transformations. Note that this method
        may not always be defined since some transformations aren't 1-1.

        Parameters
        ----------
        transformed: np.ndarray
            Array which was previously transformed by this class.
        """
        raise NotImplementedError(
            "Each Transformer is responsible for its own untransform method.")

    def transform(self,
                  dataset: Dataset,
                  parallel: bool = False,
                  out_dir: Optional[str] = None,
                  **kwargs) -> Dataset:
        """Transforms all internally stored data in dataset.

        This method transforms all internal data in the provided dataset by using
        the `Dataset.transform` method. Note that this method adds X-transform,
        y-transform columns to metadata. Specified keyword arguments are passed on
        to `Dataset.transform`.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset object to be transformed.
        parallel: bool, optional (default False)
            if True, use multiple processes to transform the dataset in parallel.
            For large datasets, this might be faster.
        out_dir: str, optional
            If `out_dir` is specified in `kwargs` and `dataset` is a `DiskDataset`,
            the output dataset will be written to the specified directory.

        Returns
        -------
        Dataset
            A newly transformed Dataset object
        """
        # Add this case in to handle non-DiskDataset that should be written to disk
        if out_dir is not None:
            if not isinstance(dataset, dc.data.DiskDataset):
                dataset = dc.data.DiskDataset.from_numpy(
                    dataset.X, dataset.y, dataset.w, dataset.ids)
        _, y_shape, w_shape, _ = dataset.get_shape()
        if y_shape == tuple() and self.transform_y:
            raise ValueError("Cannot transform y when y_values are not present")
        if w_shape == tuple() and self.transform_w:
            raise ValueError("Cannot transform w when w_values are not present")
        return dataset.transform(self, out_dir=out_dir, parallel=parallel)

    def transform_on_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transforms numpy arrays X, y, and w

        DEPRECATED. Use `transform_array` instead.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        warnings.warn(
            "transform_on_array() is deprecated and has been renamed to transform_array()."
            "transform_on_array() will be removed in DeepChem 3.0",
            FutureWarning)
        X, y, w, ids = self.transform_array(X, y, w, ids)
        return X, y, w, ids


def undo_transforms(y: np.typing.ArrayLike,
                    transformers: List[Transformer]) -> np.ndarray:
    """Undoes all transformations applied.

    Transformations are reversed using `transformer.untransform`.
    Transformations will be assumed to have been applied in the order specified,
    so transformations will be reversed in the opposite order. That is if
    `transformers = [t1, t2]`, then this method will do `t2.untransform`
    followed by `t1.untransform`.

    Parameters
    ----------
    y: np.ndarray
        Array of values for which transformations have to be undone.
    transformers: list[dc.trans.Transformer]
        List of transformations which have already been applied to `y` in the
        order specifed.

    Returns
    -------
    y_out: np.ndarray
        The array with all transformations reversed.
    """
    # Note that transformers have to be undone in reversed order
    y_out = np.asarray(y)
    for transformer in reversed(transformers):
        if transformer.transform_y:
            y_out = transformer.untransform(y_out)
    return y_out


class MinMaxTransformer(Transformer):
    """Ensure each value rests between 0 and 1 by using the min and max.

    `MinMaxTransformer` transforms the dataset by shifting each axis of X or y
    (depending on whether transform_X or transform_y is True), except the first
    one by the minimum value along the axis and dividing the result by the range
    (maximum value - minimum value) along the axis. This ensures each axis is
    between 0 and 1. In case of multi-task learning, it ensures each task is
    given equal importance.

    Given original array A, the transformed array can be written as:

    >>> import numpy as np
    >>> A = np.random.rand(10, 10)
    >>> A_min = np.min(A, axis=0)
    >>> A_max = np.max(A, axis=0)
    >>> A_t = np.nan_to_num((A - A_min)/(A_max - A_min))

    Examples
    --------

    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.random.rand(n_samples, n_tasks)
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.MinMaxTransformer(transform_y=True, dataset=dataset)
    >>> dataset = transformer.transform(dataset)

    Note
    ----
    This class can only transform `X` or `y` and not `w`. So only one of
    `transform_X` or `transform_y` can be set.

    Raises
    ------
    ValueError
        if `transform_X` and `transform_y` are both set.
    """

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 dataset: Optional[Dataset] = None):
        """Initialization of MinMax transformer.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        """
        if transform_X and transform_y:
            raise ValueError("Can only transform only one of X and y")
        if dataset is not None and transform_X:
            self.X_min = np.min(dataset.X, axis=0)
            self.X_max = np.max(dataset.X, axis=0)
        elif dataset is not None and transform_y:
            self.y_min = np.min(dataset.y, axis=0)
            self.y_max = np.max(dataset.y, axis=0)

            if len(dataset.y.shape) > 1:
                assert len(self.y_min) == dataset.y.shape[1]

        super(MinMaxTransformer, self).__init__(transform_X=transform_X,
                                                transform_y=transform_y,
                                                dataset=dataset)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of ids.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        if self.transform_X:
            # Handle division by zero
            denominator = np.where((self.X_max - self.X_min) > 0,
                                   (self.X_max - self.X_min),
                                   np.ones_like(self.X_max - self.X_min))
            X = np.nan_to_num((X - self.X_min) / denominator)
        elif self.transform_y:
            # Handle division by zero
            denominator = np.where((self.y_max - self.y_min) > 0,
                                   (self.y_max - self.y_min),
                                   np.ones_like(self.y_max - self.y_min))
            y = np.nan_to_num((y - self.y_min) / denominator)
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Undo transformation on provided data.

        Parameters
        ----------
        z: np.ndarray
            Transformed X or y array

        Returns
        -------
        np.ndarray
            Array with min-max scaling undone.
        """
        if self.transform_X:
            X_max = self.X_max
            X_min = self.X_min

            return z * (X_max - X_min) + X_min

        elif self.transform_y:
            y_min = self.y_min
            y_max = self.y_max

            n_tasks = len(y_min)
            z_shape = list(z.shape)
            z_shape.reverse()

            for dim in z_shape:
                if dim != n_tasks and dim == 1:
                    y_min = np.expand_dims(y_min, -1)
                    y_max = np.expand_dims(y_max, -1)

            y = z * (y_max - y_min) + y_min
            return y

        else:
            return z


class NormalizationTransformer(Transformer):
    """Normalizes dataset to have zero mean and unit standard deviation

    This transformer transforms datasets to have zero mean and unit standard
    deviation.

    Examples
    --------

    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.random.rand(n_samples, n_tasks)
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)
    >>> dataset = transformer.transform(dataset)

    Note
    ----
    This class can only transform `X` or `y` and not `w`. So only one of
    `transform_X` or `transform_y` can be set.

    Raises
    ------
    ValueError
        if `transform_X` and `transform_y` are both set.
    """

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 transform_w: bool = False,
                 dataset: Optional[Dataset] = None,
                 transform_gradients: bool = False,
                 move_mean: bool = True):
        """Initialize normalization transformation.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        transform_w: bool, optional (default False)
            Whether to transform w
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        """
        if transform_X and transform_y:
            raise ValueError("Can only transform only one of X and y")
        if transform_w:
            raise ValueError(
                "MinMaxTransformer doesn't support w transformation.")
        if dataset is not None and transform_X:
            X_means, X_stds = dataset.get_statistics(X_stats=True,
                                                     y_stats=False)
            self.X_means = X_means
            self.X_stds = X_stds
        elif dataset is not None and transform_y:
            y_means, y_stds = dataset.get_statistics(X_stats=False,
                                                     y_stats=True)
            self.y_means = y_means
            # Control for pathological case with no variance.
            y_stds_np = np.array(y_stds)
            y_stds_np[y_stds_np == 0] = 1.
            self.y_stds = y_stds_np
        self.transform_gradients = transform_gradients
        self.move_mean = move_mean
        if self.transform_gradients:
            true_grad, ydely_means = get_grad_statistics(dataset)
            self.grad = np.reshape(true_grad, (true_grad.shape[0], -1, 3))
            self.ydely_means = ydely_means

        super(NormalizationTransformer, self).__init__(transform_X=transform_X,
                                                       transform_y=transform_y,
                                                       transform_w=transform_w,
                                                       dataset=dataset)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of ids.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        if self.transform_X:
            if not hasattr(self, 'move_mean') or self.move_mean:
                X = np.nan_to_num((X - self.X_means) / self.X_stds)
            else:
                X = np.nan_to_num(X / self.X_stds)
        if self.transform_y:
            if not hasattr(self, 'move_mean') or self.move_mean:
                y = np.nan_to_num((y - self.y_means) / self.y_stds)
            else:
                y = np.nan_to_num(y / self.y_stds)
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Undo transformation on provided data.

        Parameters
        ----------
        z: np.ndarray
            Array to transform back

        Returns
        -------
        z_out: np.ndarray
            Array with normalization undone.
        """
        if self.transform_X:
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * self.X_stds + self.X_means
            else:
                return z * self.X_stds
        elif self.transform_y:
            y_stds = self.y_stds
            y_means = self.y_means
            # Handle case with 1 task correctly
            if len(self.y_stds.shape) == 0:
                n_tasks = 1
            else:
                n_tasks = self.y_stds.shape[0]
            z_shape = list(z.shape)
            # Get the reversed shape of z: (..., n_tasks, batch_size)
            z_shape.reverse()
            # Find the task dimension of z
            for dim in z_shape:
                if dim != n_tasks and dim == 1:
                    # Prevent broadcasting on wrong dimension
                    y_stds = np.expand_dims(y_stds, -1)
                    y_means = np.expand_dims(y_means, -1)
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * y_stds + y_means
            else:
                return z * y_stds
        else:
            return z

    def untransform_grad(self, grad, tasks):
        """DEPRECATED. DO NOT USE."""
        logger.warning(
            "NormalizationTransformer.untransform_grad is DEPRECATED and will be removed in a future version of DeepChem. "
            "Manually implement transforms to perform force calculations.")
        if self.transform_y:

            grad_means = self.y_means[1:]
            energy_var = self.y_stds[0]
            grad_var = 1 / energy_var * (self.ydely_means -
                                         self.y_means[0] * self.y_means[1:])
            energy = tasks[:, 0]
            transformed_grad = []

            for i in range(energy.size):
                Etf = energy[i]
                grad_Etf = grad[i].flatten()
                grad_E = Etf * grad_var + energy_var * grad_Etf + grad_means
                grad_E = np.reshape(grad_E, (-1, 3))
                transformed_grad.append(grad_E)

            transformed_grad = np.asarray(transformed_grad)
            return transformed_grad


class ClippingTransformer(Transformer):
    """Clip large values in datasets.

    Examples
    --------
    Let's clip values from a synthetic dataset

    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.zeros((n_samples, n_tasks))
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.ClippingTransformer(transform_X=True)
    >>> dataset = transformer.transform(dataset)
    """

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 dataset: Optional[Dataset] = None,
                 x_max: float = 5.,
                 y_max: float = 500.):
        """Initialize clipping transformation.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        dataset: dc.data.Dataset object, optional
            Dataset to be transformed
        x_max: float, optional
            Maximum absolute value for X
        y_max: float, optional
            Maximum absolute value for y

        Note
        ----
        This transformer can transform `X` and `y` jointly, but does not transform
        `w`.

        Raises
        ------
        ValueError
            if `transform_w` is set.
        """
        super(ClippingTransformer, self).__init__(transform_X=transform_X,
                                                  transform_y=transform_y,
                                                  dataset=dataset)

        self.x_max = x_max
        self.y_max = y_max

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of Features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights
        ids: np.ndarray
            Array of ids.

        Returns
        -------
        X: np.ndarray
            Transformed features
        y: np.ndarray
            Transformed tasks
        w: np.ndarray
            Transformed weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        if self.transform_X:
            X[X > self.x_max] = self.x_max
            X[X < (-1.0 * self.x_max)] = -1.0 * self.x_max
        if self.transform_y:
            y[y > self.y_max] = self.y_max
            y[y < (-1.0 * self.y_max)] = -1.0 * self.y_max
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError(
            "Cannot untransform datasets with ClippingTransformer.")


class LogTransformer(Transformer):
    """Computes a logarithmic transformation

    This transformer computes the transformation given by

    >>> import numpy as np
    >>> A = np.random.rand(10, 10)
    >>> A = np.log(A + 1)

    Assuming that tasks/features are not specified. If specified, then
    transformations are only performed on specified tasks/features.

    Examples
    --------
    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.zeros((n_samples, n_tasks))
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.LogTransformer(transform_X=True)
    >>> dataset = transformer.transform(dataset)

    Note
    ----
    This class can only transform `X` or `y` and not `w`. So only one of
    `transform_X` or `transform_y` can be set.

    Raises
    ------
    ValueError
        if `transform_w` is set or `transform_X` and `transform_y` are both set.
    """

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 features: Optional[List[int]] = None,
                 tasks: Optional[List[str]] = None,
                 dataset: Optional[Dataset] = None):
        """Initialize log transformer.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        features: list[Int]
            List of features indices to transform
        tasks: list[str]
            List of task names to transform.
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        """
        if transform_X and transform_y:
            raise ValueError("Can only transform only one of X and y")
        self.features = features
        self.tasks = tasks
        super(LogTransformer, self).__init__(transform_X=transform_X,
                                             transform_y=transform_y,
                                             dataset=dataset)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of weights.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        if self.transform_X:
            num_features = len(X[0])
            if self.features is None:
                X = np.log(X + 1)
            else:
                for j in range(num_features):
                    if j in self.features:
                        X[:, j] = np.log(X[:, j] + 1)
                    else:
                        X[:, j] = X[:, j]
        if self.transform_y:
            if np.isscalar(y[0]):
                num_tasks = 1
            else:
                num_tasks = len(y[0])
            if self.tasks is None:
                y = np.log(y + 1)
            else:
                for j in range(num_tasks):
                    if j in self.tasks:
                        y[:, j] = np.log(y[:, j] + 1)
                    else:
                        y[:, j] = y[:, j]
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Undo transformation on provided data.

        Parameters
        ----------
        z: np.ndarray,
            Transformed X or y array

        Returns
        -------
        np.ndarray
            Array with a logarithmic transformation undone.
        """
        if self.transform_X:
            num_features = len(z[0])
            if self.features is None:
                return np.exp(z) - 1
            else:
                for j in range(num_features):
                    if j in self.features:
                        z[:, j] = np.exp(z[:, j]) - 1
                    else:
                        z[:, j] = z[:, j]
                return z
        elif self.transform_y:
            if np.isscalar(z[0]):
                num_tasks = 1
            else:
                num_tasks = len(z[0])
            if self.tasks is None:
                return np.exp(z) - 1
            else:
                for j in range(num_tasks):
                    if j in self.tasks:
                        z[:, j] = np.exp(z[:, j]) - 1
                    else:
                        z[:, j] = z[:, j]
                return z
        else:
            return z


class BalancingTransformer(Transformer):
    """Balance positive and negative (or multiclass) example weights.

    This class balances the sample weights so that the sum of all example
    weights from all classes is the same. This can be useful when you're
    working on an imbalanced dataset where there are far fewer examples of some
    classes than others.

    Examples
    --------

    Here's an example for a binary dataset.

    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> n_classes = 2
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.random.randint(n_classes, size=(n_samples, n_tasks))
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> transformer = dc.trans.BalancingTransformer(dataset=dataset)
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
    >>> transformer = dc.trans.BalancingTransformer(dataset=dataset)
    >>> dataset = transformer.transform(dataset)

    See Also
    --------
    deepchem.trans.DuplicateBalancingTransformer: Balance by duplicating samples.


    Note
    ----
    This transformer is only meaningful for classification datasets where `y`
    takes on a limited set of values. This class can only transform `w` and does
    not transform `X` or `y`.

    Raises
    ------
    ValueError
        if `transform_X` or `transform_y` are set. Also raises or if `y` or `w` aren't of shape `(N,)` or `(N, n_tasks)`.
    """

    def __init__(self, dataset: Dataset):
        # BalancingTransformer can only transform weights.
        super(BalancingTransformer, self).__init__(transform_w=True,
                                                   dataset=dataset)

        # Compute weighting factors from dataset.
        y = dataset.y
        w = dataset.w
        # Handle 1-D case
        if len(y.shape) == 1:
            y = np.reshape(y, (len(y), 1))
        if len(w.shape) == 1:
            w = np.reshape(w, (len(w), 1))
        if len(y.shape) != 2:
            raise ValueError("y must be of shape (N,) or (N, n_tasks)")
        if len(w.shape) != 2:
            raise ValueError("w must be of shape (N,) or (N, n_tasks)")
        self.classes = sorted(np.unique(y))
        weights = []
        for ind, task in enumerate(dataset.get_task_names()):
            task_w = w[:, ind]
            task_y = y[:, ind]
            # Remove labels with zero weights
            task_y = task_y[task_w != 0]
            N_task = len(task_y)
            class_counts = []
            # Note that we may have 0 elements of a given class since we remove those
            # labels with zero weight. This typically happens in multitask datasets
            # where some datapoints only have labels for some tasks.
            for c in self.classes:
                # this works because task_y is 1D
                num_c = len(np.where(task_y == c)[0])
                class_counts.append(num_c)
            # This is the right ratio since N_task/num_c * num_c = N_task
            # for all classes
            class_weights = [
                N_task / float(num_c) if num_c > 0 else 0
                for num_c in class_counts
            ]
            weights.append(class_weights)
        self.weights = weights

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of weights.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        w_balanced = np.zeros_like(w)
        if len(y.shape) == 1 and len(w.shape) == 2 and w.shape[1] == 1:
            y = np.expand_dims(y, 1)
        if len(y.shape) == 1:
            n_tasks = 1
        elif len(y.shape) == 2:
            n_tasks = y.shape[1]
        else:
            raise ValueError("y must be of shape (N,) or (N, n_tasks)")
        for ind in range(n_tasks):
            if n_tasks == 1:
                task_y = y
                task_w = w
            else:
                task_y = y[:, ind]
                task_w = w[:, ind]
            for i, c in enumerate(self.classes):
                class_indices = np.logical_and(task_y == c, task_w != 0)
                # Set to the class weight computed previously
                if n_tasks == 1:
                    w_balanced[class_indices] = self.weights[ind][i]
                else:
                    w_balanced[class_indices, ind] = self.weights[ind][i]
        return (X, y, w_balanced, ids)


class FlatteningTransformer(Transformer):
    """This transformer is required for a `Dataset` consisting of fragments as a preprocessing
        step before prediction. This is used only in the context of performing interpretation of models using atomic
        contributions (atom-based model interpretation) [1]_

    Examples
    --------

    Here's an example of preparation to atom-based model interpretation.

    >>> import tempfile
    >>> import deepchem as dc
    >>> with tempfile.NamedTemporaryFile(mode='wt', delete=False) as fin:
    ...     tmp = fin.write("smiles,endpoint\\nc1ccccc1,1")
    >>> loader = dc.data.CSVLoader([], feature_field="smiles",
    ...    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False))
    >>> # prepare dataset of molecules ready for prediction stage
    ... dataset = loader.create_dataset(fin.name)

    >>> loader = dc.data.CSVLoader([], feature_field="smiles",
    ...    featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True))
    >>> frag_dataset = loader.create_dataset(fin.name)
    >>> transformer = dc.trans.FlatteningTransformer(dataset=frag_dataset)
    >>> # prepare dataset of fragments ready for prediction stage,
    ... # thereafter difference with molecules' predictions can be calculated
    ... frag_dataset = transformer.transform(frag_dataset)

    See Also
    --------
    Detailed examples of `GraphConvModel` interpretation are provided in Tutorial #28

    References
    ---------

    .. [1] Polishchuk, P., et al. J. Chem. Inf. Model. 2016, 56, 8, 1455–1469
    """

    def __init__(self, dataset: Dataset):
        """Initializes flattening transformation.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset object to be transformed
        """
        if self.__class__.__name__ == "Transformer":
            raise ValueError(
                "Transformer is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
            )
        self.transform_X = True
        self.transform_y = (dataset.get_shape()[1] != tuple()
                           )  # iff y passed, then transform it
        self.transform_w = (dataset.get_shape()[2] != tuple()
                           )  # iff w passed, then transform it
        self.transform_ids = True

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of weights.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        ids = np.repeat(ids, [len(i) for i in X],
                        axis=0)  # each fragment should recieve parent mol id
        if self.transform_y:
            y = np.repeat(
                y, [len(i) for i in X], axis=0
            )  # for consistency of shapes each fragment should recieve parent mol y
        if self.transform_w:
            w = np.repeat(
                w, [len(i) for i in X], axis=0
            )  # for consistency of shapes each fragment should recieve parent mol w
        X = np.array([j for i in X for j in i])  # flatten
        return (X, y, w, ids)


class CDFTransformer(Transformer):
    """Histograms the data and assigns values based on sorted list.

    Acts like a Cumulative Distribution Function (CDF). If given a dataset of
    samples from a continuous distribution computes the CDF of this dataset and
    replaces values with their corresponding CDF values.

    Examples
    --------
    Let's look at an example where we transform only features.

    >>> N = 10
    >>> n_feat = 5
    >>> n_bins = 100

    Note that we're using 100 bins for our CDF histogram

    >>> import numpy as np
    >>> X = np.random.normal(size=(N, n_feat))
    >>> y = np.random.randint(2, size=(N,))
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> cdftrans = dc.trans.CDFTransformer(transform_X=True, dataset=dataset, bins=n_bins)
    >>> dataset = cdftrans.transform(dataset)

    Note that you can apply this transformation to `y` as well

    >>> X = np.random.normal(size=(N, n_feat))
    >>> y = np.random.normal(size=(N,))
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> cdftrans = dc.trans.CDFTransformer(transform_y=True, dataset=dataset, bins=n_bins)
    >>> dataset = cdftrans.transform(dataset)
    """

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 dataset: Optional[Dataset] = None,
                 bins: int = 2):
        """Initialize this transformer.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        bins: int, optional (default 2)
            Number of bins to use when computing histogram.
        """
        super(CDFTransformer, self).__init__(transform_X=transform_X,
                                             transform_y=transform_y)
        self.bins = bins
        if transform_y:
            if dataset is None:
                raise ValueError(
                    "dataset must be specified when transforming y")
            self.y = dataset.y

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs CDF transform on data.

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
        idstrans: np.ndarray
            Transformed array of ids
        """
        w_t = w
        if self.transform_X:
            X_t = get_cdf_values(X, self.bins)
            y_t = y
        elif self.transform_y:
            X_t = X
            y_t = get_cdf_values(y, self.bins)
        return X_t, y_t, w_t, ids

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Undo transformation on provided data.

        Note that this transformation is only undone for y.

        Parameters
        ----------
        z: np.ndarray,
            Transformed y array

        Returns
        -------
        np.ndarray
            Array with the transformation undone.
        """
        # Need this for transform_y
        if self.transform_y:
            return self.y
        else:
            raise NotImplementedError


def get_cdf_values(array: np.ndarray, bins: int) -> np.ndarray:
    """Helper function to compute CDF values.

    Parameters
    ----------
    array: np.ndarray
        Must be of shape `(n_rows, n_cols)` or `(n_rows,)`
    bins: int
        Number of bins to split data into.

    Returns
    -------
    array_t: np.ndarray
        Array with sorted histogram values
    """
    # Handle 1D case
    if len(array.shape) == 1:
        array = np.reshape(array, (len(array), 1))
    n_rows = array.shape[0]
    n_cols = array.shape[1]
    array_t = np.zeros((n_rows, n_cols))
    parts = n_rows / bins
    hist_values = np.zeros(n_rows)
    sorted_hist_values = np.zeros(n_rows)
    for row in range(n_rows):
        if np.remainder(bins, 2) == 1:
            hist_values[row] = np.floor(np.divide(row, parts)) / (bins - 1)
        else:
            hist_values[row] = np.floor(np.divide(row, parts)) / bins
    for col in range(n_cols):
        order = np.argsort(array[:, col], axis=0)
        sorted_hist_values = hist_values[order]
        array_t[:, col] = sorted_hist_values

    return array_t


class PowerTransformer(Transformer):
    """Takes power n transforms of the data based on an input vector.

    Computes the specified powers of the dataset. This can be useful if you're
    looking to add higher order features of the form `x_i^2`, `x_i^3` etc. to
    your dataset.

    Examples
    --------
    Let's look at an example where we transform only `X`.

    >>> N = 10
    >>> n_feat = 5
    >>> powers = [1, 2, 0.5]

    So in this example, we're taking the identity, squares, and square roots.
    Now let's construct our matrices

    >>> import numpy as np
    >>> X = np.random.rand(N, n_feat)
    >>> y = np.random.normal(size=(N,))
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> trans = dc.trans.PowerTransformer(transform_X=True, dataset=dataset, powers=powers)
    >>> dataset = trans.transform(dataset)

    Let's now look at an example where we transform `y`. Note that the `y`
    transform expands out the feature dimensions of `y` the same way it does for
    `X` so this transform is only well defined for singletask datasets.

    >>> import numpy as np
    >>> X = np.random.rand(N, n_feat)
    >>> y = np.random.rand(N)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> trans = dc.trans.PowerTransformer(transform_y=True, dataset=dataset, powers=powers)
    >>> dataset = trans.transform(dataset)
    """

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 dataset: Optional[Dataset] = None,
                 powers: List[int] = [1]):
        """Initialize this transformer

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed. Note that this argument is ignored since
        `PowerTransformer` doesn't require it to be specified.
            powers: list[int], optional (default `[1]`)
            The list of powers of features/labels to compute.
        """
        super(PowerTransformer, self).__init__(transform_X=transform_X,
                                               transform_y=transform_y)
        self.powers = powers

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs power transform on data.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        if not (len(y.shape) == 1 or len(y.shape) == 2 and y.shape[1] == 1):
            raise ValueError("This transform is not defined for multitask y")
        # THis reshape is safe because of guard above.
        y = np.reshape(y, (len(y), 1))
        w_t = w
        n_powers = len(self.powers)
        if self.transform_X:
            X_t = np.power(X, self.powers[0])
            for i in range(1, n_powers):
                X_t = np.hstack((X_t, np.power(X, self.powers[i])))
            y_t = y
        if self.transform_y:
            y_t = np.power(y, self.powers[0])
            for i in range(1, n_powers):
                y_t = np.hstack((y_t, np.power(y, self.powers[i])))
            X_t = X
        return (X_t, y_t, w_t, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Undo transformation on provided data.

        Parameters
        ----------
        z: np.ndarray,
            Transformed y array

        Returns
        -------
        np.ndarray
            Array with the power transformation undone.
        """
        n_powers = len(self.powers)
        orig_len = (z.shape[1]) // n_powers
        z = z[:, :orig_len]
        z = np.power(z, 1 / self.powers[0])
        return z


class CoulombFitTransformer(Transformer):
    """Performs randomization and binarization operations on batches of Coulomb Matrix features during fit.

    Examples
    --------
    >>> n_samples = 10
    >>> n_features = 3
    >>> n_tasks = 1
    >>> ids = np.arange(n_samples)
    >>> X = np.random.rand(n_samples, n_features, n_features)
    >>> y = np.zeros((n_samples, n_tasks))
    >>> w = np.ones((n_samples, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
    >>> fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
    >>> model = dc.models.MultitaskFitTransformRegressor(n_tasks,
    ...    [n_features, n_features], batch_size=n_samples, fit_transformers=fit_transformers, n_evals=1)
    >>> print(model.n_features)
    12
    """

    def __init__(self, dataset: Dataset):
        """Initializes CoulombFitTransformer.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset object to be transformed.
        """
        X = dataset.X
        num_atoms = X.shape[1]
        self.step = 1.0
        self.noise = 1.0
        self.triuind = (np.arange(num_atoms)[:, np.newaxis] <=
                        np.arange(num_atoms)[np.newaxis, :]).flatten()
        self.max = 0
        for _ in range(10):
            self.max = np.maximum(self.max, self.realize(X).max(axis=0))
        X = self.expand(self.realize(X))
        self.nbout = X.shape[1]
        self.mean = X.mean(axis=0)
        self.std = (X - self.mean).std()
        super(CoulombFitTransformer, self).__init__(transform_X=True)

    def realize(self, X: np.ndarray) -> np.ndarray:
        """Randomize features.

        Parameters
        ----------
        X: np.ndarray
            Features

        Returns
        -------
        X: np.ndarray
            Randomized features
        """

        def _realize_(x):
            assert (len(x.shape) == 2)
            inds = np.argsort(-(x**2).sum(axis=0)**.5 +
                              np.random.normal(0, self.noise, x[0].shape))
            x = x[inds, :][:, inds] * 1
            x = x.flatten()[self.triuind]
            return x

        return np.array([_realize_(z) for z in X])

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features.

        Parameters
        ----------
        X: np.ndarray
            Features

        Returns
        -------
        X: np.ndarray
            Normalized features
        """
        return (X - self.mean) / self.std

    def expand(self, X: np.ndarray) -> np.ndarray:
        """Binarize features.

        Parameters
        ----------
        X: np.ndarray
            Features

        Returns
        -------
        X: np.ndarray
            Binarized features
        """
        Xexp = []
        for i in range(X.shape[1]):
            for k in np.arange(
                    0,
                    self.max[i] + self.step,  # type: ignore
                    self.step):
                Xexp += [np.tanh((X[:, i] - k) / self.step)]
        return np.array(Xexp).T

    def X_transform(self, X: np.ndarray) -> np.ndarray:
        """Perform Coulomb Fit transform on features.

        Parameters
        ----------
        X: np.ndarray
            Features

        Returns
        -------
        X: np.ndarray
            Transformed features
        """

        X = self.normalize(self.expand(self.realize(X)))
        return X

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs randomization and binarization operations on data.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        X = self.X_transform(X)
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        "Not implemented."
        raise NotImplementedError(
            "Cannot untransform datasets with FitTransformer.")


class IRVTransformer(Transformer):
    """Performs transform from ECFP to IRV features(K nearest neighbors).

    This transformer is required by `MultitaskIRVClassifier` as a preprocessing
    step before training.

    Examples
    --------
    Let's start by defining the parameters of the dataset we're about to
    transform.

    >>> n_feat = 128
    >>> N = 20
    >>> n_tasks = 2

    Let's now make our dataset object

    >>> import numpy as np
    >>> import deepchem as dc
    >>> X = np.random.randint(2, size=(N, n_feat))
    >>> y = np.zeros((N, n_tasks))
    >>> w = np.ones((N, n_tasks))
    >>> dataset = dc.data.NumpyDataset(X, y, w)

    And let's apply our transformer with 10 nearest neighbors.

    >>> K = 10
    >>> trans = dc.trans.IRVTransformer(K, n_tasks, dataset)
    >>> dataset = trans.transform(dataset)

    Note
    ----
    This class requires TensorFlow to be installed.
    """

    def __init__(self, K: int, n_tasks: int, dataset: Dataset):
        """Initializes IRVTransformer.

        Parameters
        ----------
        K: int
            number of nearest neighbours being count
        n_tasks: int
            number of tasks
        dataset: dc.data.Dataset object
            train_dataset
        """
        self.X = dataset.X
        self.n_tasks = n_tasks
        self.K = K
        self.y = dataset.y
        self.w = dataset.w
        super(IRVTransformer, self).__init__(transform_X=True)

    def realize(self, similarity: np.ndarray, y: np.ndarray,
                w: np.ndarray) -> List:
        """find samples with top ten similarity values in the reference dataset

        Parameters
        ----------
        similarity: np.ndarray
            similarity value between target dataset and reference dataset
            should have size of (n_samples_in_target, n_samples_in_reference)
        y: np.array
            labels for a single task
        w: np.array
            weights for a single task

        Returns
        -------
        features: list
            n_samples * np.array of size (2*K,)
            each array includes K similarity values and corresponding labels
        """
        features = []
        similarity_xs = similarity * np.sign(w)
        [target_len, reference_len] = similarity_xs.shape
        values = []
        top_labels = []
        # map the indices to labels
        for count in range(target_len // 100 + 1):
            similarity = similarity_xs[count * 100:min((count + 1) *
                                                       100, target_len), :]
            # generating batch of data by slicing similarity matrix
            # into 100*reference_dataset_length
            indice = np.argsort(similarity)[:, -(self.K + 1):][:, ::-1]
            value = np.take_along_axis(similarity, indice, axis=1)
            top_label = np.take(y, indice)
            values.append(value)
            top_labels.append(top_label)
        values_np = np.concatenate(values, axis=0)
        top_labels_np = np.concatenate(top_labels, axis=0)
        # concatenate batches of data together
        for count in range(values_np.shape[0]):
            if values_np[count, 0] == 1:
                features.append(
                    np.concatenate([
                        values_np[count, 1:(self.K + 1)],
                        top_labels_np[count, 1:(self.K + 1)]
                    ]))
                # highest similarity is 1: target is in the reference
                # use the following K points
            else:
                features.append(
                    np.concatenate([
                        values_np[count, 0:self.K], top_labels_np[count,
                                                                  0:self.K]
                    ]))
                # highest less than 1: target not in the reference, use top K points
        return features

    def X_transform(self, X_target: np.ndarray) -> np.ndarray:
        """Calculate similarity between target dataset(X_target) and
            reference dataset(X): #(1 in intersection)/#(1 in union)

        similarity = (X_target intersect X)/(X_target union X)

        Parameters
        ----------
        X_target: np.ndarray
            fingerprints of target dataset
            should have same length with X in the second axis

        Returns
        -------
        X_target: np.ndarray
            features of size(batch_size, 2*K*n_tasks)
        """
        X_target2 = []
        n_features = X_target.shape[1]
        logger.info('start similarity calculation')
        time1 = time.time()
        similarity = IRVTransformer.matrix_mul(X_target, np.transpose(
            self.X)) / (n_features - IRVTransformer.matrix_mul(
                1 - X_target, np.transpose(1 - self.X)))
        time2 = time.time()
        logger.info('similarity calculation takes %i s' % (time2 - time1))
        for i in range(self.n_tasks):
            X_target2.append(
                self.realize(similarity, self.y[:, i], self.w[:, i]))
        return np.concatenate([z for z in np.array(X_target2)], axis=1)

    @staticmethod
    def matrix_mul(X1, X2, shard_size=5000):
        """Calculate matrix multiplication for big matrix,
        X1 and X2 are sliced into pieces with shard_size rows(columns)
        then multiplied together and concatenated to the proper size
        """
        X1 = np.float_(X1)
        X2 = np.float_(X2)
        X1_shape = X1.shape
        X2_shape = X2.shape
        assert X1_shape[1] == X2_shape[0]
        X1_iter = X1_shape[0] // shard_size + 1
        X2_iter = X2_shape[1] // shard_size + 1
        all_result = np.zeros((1,))
        for X1_id in range(X1_iter):
            result = np.zeros((1,))
            for X2_id in range(X2_iter):
                partial_result = np.matmul(
                    X1[X1_id * shard_size:min((X1_id + 1) *
                                              shard_size, X1_shape[0]), :],
                    X2[:, X2_id * shard_size:min((X2_id + 1) *
                                                 shard_size, X2_shape[1])])
                # calculate matrix multiplicatin on slices
                if result.size == 1:
                    result = partial_result
                else:
                    result = np.concatenate((result, partial_result), axis=1)
                # concatenate the slices together
                del partial_result
            if all_result.size == 1:
                all_result = result
            else:
                all_result = np.concatenate((all_result, result), axis=0)
            del result
        return all_result

    def transform(self,
                  dataset: Dataset,
                  parallel: bool = False,
                  out_dir: Optional[str] = None,
                  **kwargs) -> Union[DiskDataset, NumpyDataset]:
        """Transforms a given dataset

        Parameters
        ----------
        dataset: Dataset
            Dataset to transform
        parallel: bool, optional, (default False)
            Whether to parallelize this transformation. Currently ignored.
        out_dir: str, optional (default None)
            Directory to write resulting dataset.

        Returns
        -------
        DiskDataset or NumpyDataset
        `Dataset` object that is transformed.
        """
        X_length = dataset.X.shape[0]
        X_trans = []
        for count in range(X_length // 5000 + 1):
            X_trans.append(
                self.X_transform(dataset.X[count *
                                           5000:min((count + 1) *
                                                    5000, X_length), :]))
        X = np.concatenate(X_trans, axis=0)
        if out_dir is None:
            return NumpyDataset(X, dataset.y, dataset.w, ids=None)
        return DiskDataset.from_numpy(X, dataset.y, dataset.w, data_dir=out_dir)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        "Not implemented."
        raise NotImplementedError(
            "Cannot untransform datasets with IRVTransformer.")


class DAGTransformer(Transformer):
    """Performs transform from ConvMol adjacency lists to DAG calculation orders

    This transformer is used by `DAGModel` before training to transform its
    inputs to the correct shape. This expansion turns a molecule with `n` atoms
    into `n` DAGs, each with root at a different atom in the molecule.

    Examples
    --------
    Let's transform a small dataset of molecules.

    >>> N = 10
    >>> n_feat = 5
    >>> import numpy as np
    >>> feat = dc.feat.ConvMolFeaturizer()
    >>> X = feat(["C", "CC"])
    >>> y = np.random.rand(N)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> trans = dc.trans.DAGTransformer(max_atoms=5)
    >>> dataset = trans.transform(dataset)
    """

    def __init__(self, max_atoms: int = 50):
        """Initializes DAGTransformer.

        Parameters
        ----------
        max_atoms: int, optional (Default 50)
            Maximum number of atoms to allow
        """
        self.max_atoms = max_atoms
        super(DAGTransformer, self).__init__(transform_X=True)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        for idm, mol in enumerate(X):
            X[idm].parents = self.UG_to_DAG(mol)
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        "Not implemented."
        raise NotImplementedError(
            "Cannot untransform datasets with DAGTransformer.")

    def UG_to_DAG(self, sample: ConvMol) -> List:
        """This function generates the DAGs for a molecule

        Parameters
        ----------
        sample: `ConvMol`
            Molecule to transform

        Returns
        -------
        List
            List of parent adjacency matrices
        """
        # list of calculation orders for DAGs
        # stemming from one specific atom in the molecule
        parents = []
        # starting from the adjacency list derived by graphconv featurizer
        UG = sample.get_adjacency_list()
        # number of atoms, also number of DAGs
        n_atoms = sample.get_num_atoms()
        # DAG on a molecule with k atoms includes k steps of calculation,
        # each step calculating graph features for one atom.
        # `max_atoms` is the maximum number of steps
        max_atoms = self.max_atoms
        for count in range(n_atoms):
            # each iteration generates the DAG starting from atom with index `count`
            DAG = []
            # list of lists, elements represent the calculation orders
            # for atoms in the current graph
            parent: List[Any] = [[] for i in range(n_atoms)]
            # starting from the target atom with index `count`
            current_atoms = [count]
            # flags of whether the atom is already included in the DAG
            atoms_indicator = np.zeros((n_atoms,))
            # atom `count` is in the DAG
            radial = 1
            atoms_indicator[count] = radial
            # recording number of radial propagation steps
            while not np.all(atoms_indicator):
                # in the fisrt loop, atoms directly connected to `count` will be added
                # into the DAG(radial=0), then atoms two-bond away from `count`
                # will be added in the second loop(radial=1).
                # atoms i-bond away will be added in i-th loop
                if radial > n_atoms:
                    # when molecules have separate parts, starting from one part,
                    # it is not possible to include all atoms.
                    # this break quit the loop when going into such condition
                    break
                # reinitialize targets for next iteration
                next_atoms = []
                radial = radial + 1
                for current_atom in current_atoms:
                    for atom_adj in UG[current_atom]:
                        # atoms connected to current_atom
                        if atoms_indicator[atom_adj] == 0:
                            # generate the dependency map of current DAG
                            # atoms connected to `current_atoms`(and not included in the DAG)
                            # are added, and will be the `current_atoms` for next iteration.
                            DAG.append((current_atom, atom_adj))
                            atoms_indicator[atom_adj] = radial
                            next_atoms.append(atom_adj)
                current_atoms = next_atoms
            # DAG starts from the target atom, calculation should go in reverse
            for edge in reversed(DAG):
                # `edge[1]` is the parent of `edge[0]`
                parent[edge[0]].append(edge[1] % max_atoms)
                parent[edge[0]].extend(parent[edge[1]])

            for i, order in enumerate(parent):
                parent[i] = sorted(order, key=lambda x: atoms_indicator[x])
            # after this loop, `parents[i]` includes all parents of atom i
            for ids, atom in enumerate(parent):
                # manually adding the atom index into its parents list
                parent[ids].insert(0, ids % max_atoms)
            # after this loop, `parents[i][0]` is i, `parents[i][1:]` are all parents of atom i

            # atoms with less parents(farther from the target atom) come first.
            # graph features of atoms without parents will be first calculated,
            # then atoms with more parents can be calculated in order
            # based on previously calculated graph features.
            # target atom of this DAG will be calculated in the last step
            parent = sorted(parent, key=len)

            for ids, atom in enumerate(parent):
                n_par = len(atom)
                # padding with `max_atoms`
                if n_par < max_atoms:
                    parent[ids].extend(
                        [max_atoms for i in range(max_atoms - n_par)])
                if n_par > max_atoms:
                    parent[ids] = parent[ids][:max_atoms]

            if len(parent) > max_atoms:
                parent = parent[-max_atoms:]
            while len(parent) < max_atoms:
                # padding
                parent.insert(0, [max_atoms] * max_atoms)
            # `parents[i]` is the calculation order for the DAG stemming from atom i,
            # which is a max_atoms * max_atoms numpy array after padding
            parents.append(np.array(parent))

        return parents


class ImageTransformer(Transformer):
    """Transforms images to a specified width and/or height.

    Images of shape (n_samples, width, height) and (n_samples, width, height, channels) are supported.

    Images of shape (n_samples, width, height, channels) can be resized to
    (n_samples, new_width, new_height, channels).

    Note
    ----
    This class require Pillow to be installed.
    """

    def __init__(self,
                 size: Tuple[int, int],
                 transform_X: bool = True,
                 transform_y: bool = False):
        """
        Initializes ImageTransformer.

        Parameters
        ----------
        size: Tuple[int, int]
            The image size, a tuple of (width, height).
        transform_X: bool, optional (default True)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y

        Examples
        --------
        Let's transform a small dataset of images and their masks.

        >>> import deepchem as dc
        >>> import numpy as np
        >>> X = np.random.rand(10, 256, 256, 3)
        >>> y = np.random.rand(10, 256, 256, 3)

        Let's now make a ImageDataset
        >>> dataset = dc.data.ImageDataset(X, y)

        And let's apply our transformer with a size of (128, 128, 3).
        >>> img_transform = dc.trans.ImageTransformer(size=(128, 128), transform_X=True, transform_y=True)
        >>> resized_dataset = dataset.transform(img_transform)

        We can see that our dataset has been resized.
        >>> resized_X = resized_dataset.X
        >>> resized_X.shape
        (10, 128, 128, 3)

        We can also see that the masks have been resized.
        If you want to transform only X, you can set `transform_y` to `False`, and vice versa.
        """
        self.size = size
        super(ImageTransformer, self).__init__(transform_X=transform_X,
                                               transform_y=transform_y)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        try:
            from PIL import Image
        except ModuleNotFoundError:
            raise ImportError("This function requires Pillow to be installed.")

        if self.transform_X:
            assert len(
                X.shape
            ) >= 3, "X must be an array of images with shape (n_samples, width, height) or (n_samples, width, height, channels)."
            # PIL only accepts uint8 data type as inputs, so we multiply then divide by 255 to minimize information loss while resizing.
            x = np.array([
                np.array(
                    Image.fromarray(
                        (img * 255).astype(np.uint8)).resize(self.size)) / 255
                for img in X
            ])
        else:  # if not transforming X, return the original X
            x = np.array(X)
        if self.transform_y:
            assert len(
                y.shape
            ) >= 3, "y must be an array of images with shape (n_samples, width, height) or (n_samples, width, height, channels)."
            y = np.array([
                np.array(
                    Image.fromarray(
                        (img * 255).astype(np.uint8)).resize(self.size)) / 255
                for img in y
            ])
        return (x, y, w, ids)


# class ANITransformer(Transformer):
#   """Performs transform from 3D coordinates to ANI symmetry functions

#   Note
#   ----
#   This class requires TensorFlow to be installed.
#   """

#   def __init__(self,
#                max_atoms=23,
#                radial_cutoff=4.6,
#                angular_cutoff=3.1,
#                radial_length=32,
#                angular_length=8,
#                atom_cases=[1, 6, 7, 8, 16],
#                atomic_number_differentiated=True,
#                coordinates_in_bohr=True):
#     """
#     Only X can be transformed
#     """
#     import tensorflow as tf
#     self.max_atoms = max_atoms
#     self.radial_cutoff = radial_cutoff
#     self.angular_cutoff = angular_cutoff
#     self.radial_length = radial_length
#     self.angular_length = angular_length
#     self.atom_cases = atom_cases
#     self.atomic_number_differentiated = atomic_number_differentiated
#     self.coordinates_in_bohr = coordinates_in_bohr
#     self.compute_graph = self.build()
#     self.sess = tf.Session(graph=self.compute_graph)
#     self.transform_batch_size = 32
#     super(ANITransformer, self).__init__(transform_X=True)

#   def transform_array(self, X, y, w):
#     if self.transform_X:

#       X_out = []
#       num_transformed = 0
#       start = 0

#       batch_size = self.transform_batch_size

#       while True:
#         end = min((start + 1) * batch_size, X.shape[0])
#         X_batch = X[(start * batch_size):end]
#         output = self.sess.run(
#             [self.outputs], feed_dict={self.inputs: X_batch})[0]
#         X_out.append(output)
#         num_transformed = num_transformed + X_batch.shape[0]
#         logger.info('%i samples transformed' % num_transformed)
#         start += 1
#         if end >= len(X):
#           break

#       X_new = np.concatenate(X_out, axis=0)
#       assert X_new.shape[0] == X.shape[0]
#     return (X_new, y, w)

#   def untransform(self, z):
#     raise NotImplementedError(
#         "Cannot untransform datasets with ANITransformer.")

#   def build(self):
#     """ tensorflow computation graph for transform """
#     import tensorflow as tf
#     graph = tf.Graph()
#     with graph.as_default():
#       self.inputs = tf.keras.Input(
#           dtype=tf.float32, shape=(None, self.max_atoms, 4))
#       atom_numbers = tf.cast(self.inputs[:, :, 0], tf.int32)
#       flags = tf.sign(atom_numbers)
#       flags = tf.cast(
#           tf.expand_dims(flags, 1) * tf.expand_dims(flags, 2), tf.float32)
#       coordinates = self.inputs[:, :, 1:]
#       if self.coordinates_in_bohr:
#         coordinates = coordinates * 0.52917721092
#       d = self.distance_matrix(coordinates, flags)
#       d_radial_cutoff = self.distance_cutoff(d, self.radial_cutoff, flags)
#       d_angular_cutoff = self.distance_cutoff(d, self.angular_cutoff, flags)
#       radial_sym = self.radial_symmetry(d_radial_cutoff, d, atom_numbers)
#       angular_sym = self.angular_symmetry(d_angular_cutoff, d, atom_numbers,
#                                           coordinates)
#       self.outputs = tf.concat(
#           [
#               tf.cast(tf.expand_dims(atom_numbers, 2), tf.float32), radial_sym,
#               angular_sym
#           ],
#           axis=2)
#     return graph

#   def distance_matrix(self, coordinates, flags):
#     """ Generate distance matrix """
#     import tensorflow as tf
#     max_atoms = self.max_atoms
#     tensor1 = tf.stack([coordinates] * max_atoms, axis=1)
#     tensor2 = tf.stack([coordinates] * max_atoms, axis=2)

#     # Calculate pairwise distance
#     d = tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=3))
#     # Masking for valid atom index
#     d = d * flags
#     return d

#   def distance_cutoff(self, d, cutoff, flags):
#     """ Generate distance matrix with trainable cutoff """
#     import tensorflow as tf
#     # Cutoff with threshold Rc
#     d_flag = flags * tf.sign(cutoff - d)
#     d_flag = tf.nn.relu(d_flag)
#     d_flag = d_flag * tf.expand_dims(
#         tf.expand_dims((1 - tf.eye(self.max_atoms)), 0), -1)
#     d = 0.5 * (tf.cos(np.pi * d / cutoff) + 1)
#     return d * d_flag

#   def radial_symmetry(self, d_cutoff, d, atom_numbers):
#     """ Radial Symmetry Function """
#     import tensorflow as tf
#     embedding = tf.eye(np.max(self.atom_cases) + 1)
#     atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

#     Rs = np.linspace(0., self.radial_cutoff, self.radial_length)
#     ita = np.ones_like(Rs) * 3 / (Rs[1] - Rs[0])**2
#     Rs = tf.cast(np.reshape(Rs, (1, 1, 1, -1)), tf.float32)
#     ita = tf.cast(np.reshape(ita, (1, 1, 1, -1)), tf.float32)
#     length = ita.get_shape().as_list()[-1]

#     d_cutoff = tf.stack([d_cutoff] * length, axis=3)
#     d = tf.stack([d] * length, axis=3)

#     out = tf.exp(-ita * tf.square(d - Rs)) * d_cutoff
#     if self.atomic_number_differentiated:
#       out_tensors = []
#       for atom_type in self.atom_cases:
#         selected_atoms = tf.expand_dims(
#             tf.expand_dims(atom_numbers_embedded[:, :, atom_type], axis=1),
#             axis=3)
#         out_tensors.append(tf.reduce_sum(out * selected_atoms, axis=2))
#       return tf.concat(out_tensors, axis=2)
#     else:
#       return tf.reduce_sum(out, axis=2)

#   def angular_symmetry(self, d_cutoff, d, atom_numbers, coordinates):
#     """ Angular Symmetry Function """
#     import tensorflow as tf
#     max_atoms = self.max_atoms
#     embedding = tf.eye(np.max(self.atom_cases) + 1)
#     atom_numbers_embedded = tf.nn.embedding_lookup(embedding, atom_numbers)

#     Rs = np.linspace(0., self.angular_cutoff, self.angular_length)
#     ita = 3 / (Rs[1] - Rs[0])**2
#     thetas = np.linspace(0., np.pi, self.angular_length)
#     zeta = float(self.angular_length**2)

#     ita, zeta, Rs, thetas = np.meshgrid(ita, zeta, Rs, thetas)
#     zeta = tf.cast(np.reshape(zeta, (1, 1, 1, 1, -1)), tf.float32)
#     ita = tf.cast(np.reshape(ita, (1, 1, 1, 1, -1)), tf.float32)
#     Rs = tf.cast(np.reshape(Rs, (1, 1, 1, 1, -1)), tf.float32)
#     thetas = tf.cast(np.reshape(thetas, (1, 1, 1, 1, -1)), tf.float32)
#     length = zeta.get_shape().as_list()[-1]

#     vector_distances = tf.stack([coordinates] * max_atoms, 1) - tf.stack(
#         [coordinates] * max_atoms, 2)
#     R_ij = tf.stack([d] * max_atoms, axis=3)
#     R_ik = tf.stack([d] * max_atoms, axis=2)
#     f_R_ij = tf.stack([d_cutoff] * max_atoms, axis=3)
#     f_R_ik = tf.stack([d_cutoff] * max_atoms, axis=2)

#     # Define angle theta = arccos(R_ij(Vector) dot R_ik(Vector)/R_ij(distance)/R_ik(distance))
#     vector_mul = tf.reduce_sum(tf.stack([vector_distances] * max_atoms, axis=3) * \
#                                tf.stack([vector_distances] * max_atoms, axis=2), axis=4)
#     vector_mul = vector_mul * tf.sign(f_R_ij) * tf.sign(f_R_ik)
#     theta = tf.acos(tf.math.divide(vector_mul, R_ij * R_ik + 1e-5))

#     R_ij = tf.stack([R_ij] * length, axis=4)
#     R_ik = tf.stack([R_ik] * length, axis=4)
#     f_R_ij = tf.stack([f_R_ij] * length, axis=4)
#     f_R_ik = tf.stack([f_R_ik] * length, axis=4)
#     theta = tf.stack([theta] * length, axis=4)

#     out_tensor = tf.pow((1. + tf.cos(theta - thetas)) / 2., zeta) * \
#                  tf.exp(-ita * tf.square((R_ij + R_ik) / 2. - Rs)) * f_R_ij * f_R_ik * 2

#     if self.atomic_number_differentiated:
#       out_tensors = []
#       for id_j, atom_type_j in enumerate(self.atom_cases):
#         for atom_type_k in self.atom_cases[id_j:]:
#           selected_atoms = tf.stack([atom_numbers_embedded[:, :, atom_type_j]] * max_atoms, axis=2) * \
#                            tf.stack([atom_numbers_embedded[:, :, atom_type_k]] * max_atoms, axis=1)
#           selected_atoms = tf.expand_dims(
#               tf.expand_dims(selected_atoms, axis=1), axis=4)
#           out_tensors.append(
#               tf.reduce_sum(out_tensor * selected_atoms, axis=(2, 3)))
#       return tf.concat(out_tensors, axis=2)
#     else:
#       return tf.reduce_sum(out_tensor, axis=(2, 3))

#   def get_num_feats(self):
#     n_feat = self.outputs.get_shape().as_list()[-1]
#     return n_feat


class FeaturizationTransformer(Transformer):
    """A transformer which runs a featurizer over the X values of a dataset.

    Datasets used by this transformer must be compatible with the internal
    featurizer. The idea of this transformer is that it allows for the
    application of a featurizer to an existing dataset.

    Examples
    --------
    >>> smiles = ["C", "CC"]
    >>> X = np.array(smiles)
    >>> y = np.array([1, 0])
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> trans = dc.trans.FeaturizationTransformer(dataset, dc.feat.CircularFingerprint())
    >>> dataset = trans.transform(dataset)
    """

    def __init__(self,
                 dataset: Optional[Dataset] = None,
                 featurizer: Optional[Featurizer] = None):
        """Initialization of FeaturizationTransformer

        Parameters
        ----------
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        featurizer: dc.feat.Featurizer object, optional (default None)
            Featurizer applied to perform transformations.
        """
        if featurizer is None:
            raise ValueError("featurizer must be specified.")
        self.featurizer = featurizer
        super(FeaturizationTransformer, self).__init__(transform_X=True,
                                                       dataset=dataset)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transforms arrays of rdkit mols using internal featurizer.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        X = self.featurizer.featurize(X)
        return X, y, w, ids


class DataTransforms(object):
    """Applies different data transforms to images.

    This utility class facilitates various image transformations that may be of
    use for handling image datasets.

    Note
    ----
    This class requires PIL to be installed.
    """

    def __init__(self, Image):
        self.Image = Image

    def scale(self, h, w):
        """Scales the image

        Parameters
        ----------
        h: int
            Height of the images
        w: int
            Width of the images

        Returns
        -------
        np.ndarray
            The scaled image.
        """
        from PIL import Image
        return Image.fromarray(self.Image).resize((h, w))

    def flip(self, direction="lr"):
        """Flips the image

        Parameters
        ----------
        direction: str
            "lr" denotes left-right flip and "ud" denotes up-down flip.

        Returns
        -------
        np.ndarray
            The flipped image.
        """
        if direction == "lr":
            return np.fliplr(self.Image)
        elif direction == "ud":
            return np.flipud(self.Image)
        else:
            raise ValueError(
                "Invalid flip command : Enter either lr (for left to right flip) or ud (for up to down flip)"
            )

    def rotate(self, angle=0):
        """Rotates the image

        Parameters
        ----------
        angle: float (default = 0 i.e no rotation)
            Denotes angle by which the image should be rotated (in Degrees)

        Returns
        -------
        np.ndarray
            The rotated image.
        """
        return scipy.ndimage.rotate(self.Image, angle)

    def gaussian_blur(self, sigma=0.2):
        """Adds gaussian noise to the image

        Parameters
        ----------
        sigma: float
            Std dev. of the gaussian distribution

        Returns
        -------
        np.ndarray
            The image added gaussian noise.
        """
        return scipy.ndimage.gaussian_filter(self.Image, sigma)

    def center_crop(self, x_crop, y_crop):
        """Crops the image from the center

        Parameters
        ----------
        x_crop: int
            the total number of pixels to remove in the horizontal direction, evenly split between the left and right sides
        y_crop: int
            the total number of pixels to remove in the vertical direction, evenly split between the top and bottom sides

        Returns
        -------
        np.ndarray
            The center cropped image.
        """
        y = self.Image.shape[0]
        x = self.Image.shape[1]
        x_start = x // 2 - (x_crop // 2)
        y_start = y // 2 - (y_crop // 2)
        return self.Image[y_start:y_start + y_crop, x_start:x_start + x_crop]

    def crop(self, left, top, right, bottom):
        """Crops the image and returns the specified rectangular region from an image

        Parameters
        ----------
        left: int
            the number of pixels to exclude from the left of the image
        top: int
            the number of pixels to exclude from the top of the image
        right: int
            the number of pixels to exclude from the right of the image
        bottom: int
            the number of pixels to exclude from the bottom of the image

        Returns
        -------
        np.ndarray
            The cropped image.
        """
        y = self.Image.shape[0]
        x = self.Image.shape[1]
        return self.Image[top:y - bottom, left:x - right]

    def convert2gray(self):
        """Converts the image to grayscale. The coefficients correspond to the Y' component of the Y'UV color system.

        Returns
        -------
        np.ndarray
            The grayscale image.
        """
        return np.dot(self.Image[..., :3], [0.2989, 0.5870, 0.1140])

    def shift(self, width, height, mode='constant', order=3):
        """Shifts the image

        Parameters
        ----------
        width: float
            Amount of width shift (positive values shift image right )
        height: float
            Amount of height shift(positive values shift image lower)
        mode: str
            Points outside the boundaries of the input are filled according to the
            given mode: (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is
            ‘constant’
        order: int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        Returns
        -------
        np.ndarray
            The shifted image.
        """
        if len(self.Image.shape) == 2:
            return scipy.ndimage.shift(self.Image, [height, width],
                                       order=order,
                                       mode=mode)
        if len(self.Image.shape == 3):
            return scipy.ndimage.shift(self.Image, [height, width, 0],
                                       order=order,
                                       mode=mode)

    def gaussian_noise(self, mean=0, std=25.5):
        """Adds gaussian noise to the image

        Parameters
        ----------
        mean: float
            Mean of gaussian.
        std: float
            Standard deviation of gaussian.

        Returns
        -------
        np.ndarray
            The image added gaussian noise.
        """

        x = self.Image
        x = x + np.random.normal(loc=mean, scale=std, size=self.Image.shape)
        return x

    def salt_pepper_noise(self, prob=0.05, salt=255, pepper=0):
        """Adds salt and pepper noise to the image

        Parameters
        ----------
        prob: float
            probability of the noise.
        salt: float
            value of salt noise.
        pepper: float
            value of pepper noise.

        Returns
        -------
        np.ndarray
            The image added salt and pepper noise.
        """

        noise = np.random.random(size=self.Image.shape)
        x = self.Image
        x[noise < (prob / 2)] = pepper
        x[noise > (1 - prob / 2)] = salt
        return x

    def median_filter(self, size):
        """ Calculates a multidimensional median filter

        Parameters
        ----------
        size: int
            The kernel size in pixels.

        Returns
        -------
        np.ndarray
            The median filtered image.
        """
        from PIL import Image, ImageFilter
        image = Image.fromarray(self.Image)
        image = image.filter(ImageFilter.MedianFilter(size=size))
        return np.array(image)


class RxnSplitTransformer(Transformer):
    """Splits the reaction SMILES input into the source and target strings
    required for machine translation tasks.

    The input is expected to be in the form reactant>reagent>product. The source
    string would be reactants>reagents and the target string would be the products.

    The transformer can also separate the reagents from the reactants for a mixed
    training mode. During mixed training, the source string is transformed from
    reactants>reagent to reactants.reagent> . This can be toggled (default True)
    by setting the value of sep_reagent while calling the transformer.

    Examples
    --------
    >>> # When mixed training is toggled.
    >>> import numpy as np
    >>> from deepchem.trans.transformers import RxnSplitTransformer
    >>> reactions = np.array(["CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1>C1CCOC1.[Cl-]>CC(C)CC(=O)c1ccc(O)nc1","CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>>CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21"], dtype=object)
    >>> trans = RxnSplitTransformer(sep_reagent=True)
    >>> split_reactions = trans.transform_array(X=reactions, y=np.array([]), w=np.array([]), ids=np.array([]))
    >>> split_reactions
    (array([['CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1>C1CCOC1.[Cl-]',
            'CC(C)CC(=O)c1ccc(O)nc1'],
           ['CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>',
            'CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21']], dtype='<U51'), array([], dtype=float64), array([], dtype=float64), array([], dtype=float64))

    When mixed training is disabled, you get the following outputs:

    >>> trans_disable = RxnSplitTransformer(sep_reagent=False)
    >>> split_reactions = trans_disable.transform_array(X=reactions, y=np.array([]), w=np.array([]), ids=np.array([]))
    >>> split_reactions
    (array([['CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1.C1CCOC1.[Cl-]>',
            'CC(C)CC(=O)c1ccc(O)nc1'],
           ['CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>',
            'CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21']], dtype='<U51'), array([], dtype=float64), array([], dtype=float64), array([], dtype=float64))

    Note
    ----
    This class only transforms the feature field of a reaction dataset like USPTO.
    """

    def __init__(self,
                 sep_reagent: bool = True,
                 dataset: Optional[Dataset] = None):
        """Initializes the Reaction split Transformer.

        Parameters
        ----------
        sep_reagent: bool, optional (default True)
            To separate the reagent and reactants for training.
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed.
        """

        self.sep_reagent = sep_reagent
        super(RxnSplitTransformer, self).__init__(transform_X=True,
                                                  dataset=dataset)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features(the reactions)
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of weights.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """

        reactant = list(map(lambda x: x.split('>')[0], X))
        reagent = list(map(lambda x: x.split('>')[1], X))
        product = list(map(lambda x: x.split('>')[2], X))

        if self.sep_reagent:
            source = [x + '>' + y for x, y in zip(reactant, reagent)]
        else:
            source = [
                x + '.' + y + '>' if y else x + '>' + y
                for x, y in zip(reactant, reagent)
            ]

        target = product

        X = np.column_stack((source, target))

        return (X, y, w, ids)

    def untransform(self, z):
        """Not Implemented."""
        raise NotImplementedError("Cannot untransform the source/target split.")
