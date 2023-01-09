"""
Sample supports from datasets.
"""
import logging
import time
import numpy as np
from deepchem.data import NumpyDataset

logger = logging.getLogger(__name__)


def remove_dead_examples(dataset):
    """Removes compounds with no weight.

    Parameters
    ----------
    dataset: dc.data.Dataset
        Source dataset.
    """
    w = dataset.w
    nonzero_inds = np.nonzero(np.sum(w, axis=1))

    # Remove support indices
    X = dataset.X[nonzero_inds]
    y = dataset.y[nonzero_inds]
    w = dataset.w[nonzero_inds]
    ids = dataset.ids[nonzero_inds]

    return NumpyDataset(X, y, w, ids)


def dataset_difference(dataset, remove):
    """Removes the compounds in remove from dataset.

    Parameters
    ----------
    dataset: dc.data.Dataset
        Source dataset.
    remove: dc.data.Dataset
        Dataset whose overlap will be removed.
    """
    remove_ids = set(remove.ids)
    keep_inds = [
        ind for ind in range(len(dataset)) if dataset.ids[ind] not in remove_ids
    ]

    # Remove support indices
    X = dataset.X[keep_inds]
    y = dataset.y[keep_inds]
    w = dataset.w[keep_inds]
    ids = dataset.ids[keep_inds]

    return NumpyDataset(X, y, w, ids)


def get_task_dataset_minus_support(dataset, support, task):
    """Gets data for specified task, minus support points.

    Useful for evaluating model performance once trained (so that
    test compounds can be ensured distinct from support.)

    Parameters
    ----------
    dataset: dc.data.Dataset
        Source dataset.
    support: dc.data.Dataset
        The support dataset
    task: int
        Task number of task to select.
    """
    support_ids = set(support.ids)
    non_support_inds = [
        ind for ind in range(len(dataset))
        if dataset.ids[ind] not in support_ids
    ]

    # Remove support indices
    X = dataset.X[non_support_inds]
    y = dataset.y[non_support_inds]
    w = dataset.w[non_support_inds]
    ids = dataset.ids[non_support_inds]

    # Get task specific entries
    w_task = w[:, task]
    X_task = X[w_task != 0]
    y_task = np.expand_dims(y[w_task != 0, task], 1)
    ids_task = ids[w_task != 0]
    # Now just get weights for this task
    w_task = np.expand_dims(w[w_task != 0, task], 1)

    return NumpyDataset(X_task, y_task, w_task, ids_task)


def get_task_dataset(dataset, task):
    """Selects out entries for a particular task."""
    X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids
    # Get task specific entries
    w_task = w[:, task]
    X_task = X[w_task != 0]
    y_task = np.expand_dims(y[w_task != 0, task], 1)
    ids_task = ids[w_task != 0]
    # Now just get weights for this task
    w_task = np.expand_dims(w[w_task != 0, task], 1)

    return NumpyDataset(X_task, y_task, w_task, ids_task)


def get_task_test(dataset, n_episodes, n_test, task, log_every_n=50):
    """Gets test set from specified task.

    Parameters
    ----------
    dataset: dc.data.Dataset
        Dataset from which to sample.
    n_episodes: int
        Number of episodes to sample test sets for.
    n_test: int
        Number of compounds per test set.
    log_every_n: int, optional
        Prints every log_every_n supports sampled.
    """
    w_task = dataset.w[:, task]
    X_task = dataset.X[w_task != 0]
    y_task = dataset.y[w_task != 0]
    ids_task = dataset.ids[w_task != 0]
    # Now just get weights for this task
    w_task = dataset.w[w_task != 0]

    n_samples = len(X_task)

    ids = np.random.choice(np.arange(n_samples), (n_episodes, n_test))

    tests = []
    for episode in range(n_episodes):
        if episode % log_every_n == 0:
            logger.info("Sampling test %d" % episode)
        inds = ids[episode]
        X_batch = X_task[inds]
        y_batch = np.squeeze(y_task[inds, task])
        w_batch = np.squeeze(w_task[inds, task])
        ids_batch = ids_task[inds]
        tests.append(NumpyDataset(X_batch, y_batch, w_batch, ids_batch))
    return tests


def get_single_task_test(dataset, batch_size, task, replace=True):
    """Gets test set from specified task.

    Samples random subset of size batch_size from specified task of dataset.
    Ensures that sampled points have measurements for this task.
    """
    w_task = dataset.w[:, task]
    X_task = dataset.X[w_task != 0]
    y_task = dataset.y[w_task != 0]
    ids_task = dataset.ids[w_task != 0]
    # Now just get weights for this task
    w_task = dataset.w[w_task != 0]

    inds = np.random.choice(np.arange(len(X_task)), batch_size, replace=replace)
    X_batch = X_task[inds]
    y_batch = np.squeeze(y_task[inds, task])
    w_batch = np.squeeze(w_task[inds, task])
    ids_batch = ids_task[inds]
    return NumpyDataset(X_batch, y_batch, w_batch, ids_batch)


def get_single_task_support(dataset, n_pos, n_neg, task, replace=True):
    """Generates one support set purely for specified task.

    Parameters
    ----------
    datasets: dc.data.Dataset
        Dataset from which supports are sampled.
    n_pos: int
        Number of positive samples in support.
    n_neg: int
        Number of negative samples in support.
    task: int
        Index of current task.
    replace: bool, optional
        Whether or not to use replacement when sampling supports.

    Returns
    -------
    list
        List of NumpyDatasets, each of which is a support set.
    """
    return get_task_support(dataset, 1, n_pos, n_neg, task)[0]


def get_task_support(dataset, n_episodes, n_pos, n_neg, task, log_every_n=50):
    """Generates one support set purely for specified task.

    Parameters
    ----------
    datasets: dc.data.Dataset
        Dataset from which supports are sampled.
    n_episodes: int
        Number of episodes for which supports have to be sampled from this task.
    n_pos: int
        Number of positive samples in support.
    n_neg: int
        Number of negative samples in support.
    task: int
        Index of current task.
    log_every_n: int, optional
        Prints every log_every_n supports sampled.

    Returns
    -------
    list
        List of NumpyDatasets, each of which is a support set.
    """
    y_task = dataset.y[:, task]
    w_task = dataset.w[:, task]

    # Split data into pos and neg lists.
    pos_mols = np.where(np.logical_and(y_task == 1, w_task != 0))[0]
    neg_mols = np.where(np.logical_and(y_task == 0, w_task != 0))[0]

    supports = []
    for episode in range(n_episodes):
        if episode % log_every_n == 0:
            logger.info("Sampling support %d" % episode)
        # No replacement allowed for supports
        pos_ids = np.random.choice(len(pos_mols), (n_pos,), replace=False)
        neg_ids = np.random.choice(len(neg_mols), (n_neg,), replace=False)
        pos_inds, neg_inds = pos_mols[pos_ids], neg_mols[neg_ids]
        # Handle one-d vs. non one-d feature matrices
        one_dimensional_features = (len(dataset.X.shape) == 1)
        if not one_dimensional_features:
            X = np.vstack([dataset.X[pos_inds], dataset.X[neg_inds]])
        else:
            X = np.concatenate([dataset.X[pos_inds], dataset.X[neg_inds]])
        y = np.expand_dims(
            np.concatenate(
                [dataset.y[pos_inds, task], dataset.y[neg_inds, task]]), 1)
        w = np.expand_dims(
            np.concatenate(
                [dataset.w[pos_inds, task], dataset.w[neg_inds, task]]), 1)
        ids = np.concatenate([dataset.ids[pos_inds], dataset.ids[neg_inds]])
        supports.append(NumpyDataset(X, y, w, ids))
    return supports


class EpisodeGenerator(object):
    """Generates (support, test) pairs for episodic training.

    Precomputes all (support, test) pairs at construction. Allows to reduce
    overhead from computation.
    """

    def __init__(self, dataset, n_pos, n_neg, n_test, n_episodes_per_task):
        """
        Parameters
        ----------
        dataset: dc.data.Dataset
            Holds dataset from which support sets will be sampled.
        n_pos: int
            Number of positive samples
        n_neg: int
            Number of negative samples.
        n_test: int
            Number of samples in test set.
        n_episodes_per_task: int
            Number of (support, task) pairs to sample per task.
        replace: bool
            Whether to use sampling with or without replacement.
        """
        time_start = time.time()
        self.tasks = range(len(dataset.get_task_names()))
        self.n_tasks = len(self.tasks)
        self.n_episodes_per_task = n_episodes_per_task
        self.dataset = dataset
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.task_episodes = {}

        for task in range(self.n_tasks):
            task_supports = get_task_support(self.dataset, n_episodes_per_task,
                                             n_pos, n_neg, task)
            task_tests = get_task_test(self.dataset, n_episodes_per_task,
                                       n_test, task)
            self.task_episodes[task] = (task_supports, task_tests)

        # Init the iterator
        self.perm_tasks = np.random.permutation(self.tasks)
        # Set initial iterator state
        self.task_num = 0
        self.trial_num = 0
        time_end = time.time()
        logger.info("Constructing EpisodeGenerator took %s seconds" %
                    str(time_end - time_start))

    def __iter__(self):
        return self

    def next(self):
        """Sample next (support, test) pair.

        Return from internal storage.
        """
        if self.trial_num == self.n_episodes_per_task:
            raise StopIteration
        else:
            task = self.perm_tasks[self.task_num]  # Get id from permutation
            # support = self.supports[task][self.trial_num]
            task_supports, task_tests = self.task_episodes[task]
            support, test = (task_supports[self.trial_num],
                             task_tests[self.trial_num])
            # Increment and update logic
            self.task_num += 1
            if self.task_num == self.n_tasks:
                self.task_num = 0  # Reset
                self.perm_tasks = np.random.permutation(
                    self.tasks)  # Permute again
                self.trial_num += 1  # Upgrade trial index

            return (task, support, test)

    __next__ = next  # Python 3.X compatibility


class SupportGenerator(object):
    """Generate support sets from a dataset.

    Iterates over tasks and trials. For each trial, picks one support from
    each task, and returns in a randomized order
    """

    def __init__(self, dataset, n_pos, n_neg, n_trials):
        """
        Parameters
        ----------
        dataset: dc.data.Dataset
            Holds dataset from which support sets will be sampled.
        n_pos: int
            Number of positive samples
        n_neg: int
            Number of negative samples.
        n_trials: int
            Number of passes over dataset to make. In total, n_tasks*n_trials
            support sets will be sampled by algorithm.
        """

        self.tasks = range(len(dataset.get_task_names()))
        self.n_tasks = len(self.tasks)
        self.n_trials = n_trials
        self.dataset = dataset
        self.n_pos = n_pos
        self.n_neg = n_neg

        # Init the iterator
        self.perm_tasks = np.random.permutation(self.tasks)
        # Set initial iterator state
        self.task_num = 0
        self.trial_num = 0

    def __iter__(self):
        return self

    def next(self):
        """Sample next support.

        Supports are sampled from the tasks in a random order. Each support is
        drawn entirely from within one task.
        """
        if self.trial_num == self.n_trials:
            raise StopIteration
        else:
            task = self.perm_tasks[self.task_num]  # Get id from permutation
            # support = self.supports[task][self.trial_num]
            support = get_single_task_support(self.dataset,
                                              n_pos=self.n_pos,
                                              n_neg=self.n_neg,
                                              task=task,
                                              replace=False)
            # Increment and update logic
            self.task_num += 1
            if self.task_num == self.n_tasks:
                self.task_num = 0  # Reset
                self.perm_tasks = np.random.permutation(
                    self.tasks)  # Permute again
                self.trial_num += 1  # Upgrade trial index

            return (task, support)

    __next__ = next  # Python 3.X compatibility
