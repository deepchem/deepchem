"""
Asymmetric Validation Embedding (AVE) Splitter
"""
import numpy as np
import random
import logging
from typing import Tuple, List, Optional
from deepchem.data import Dataset
from deepchem.splits.splitters import Splitter

logger = logging.getLogger(__name__)


class AVESplitter(Splitter):
    """
    Splitter that minimizes Asymmetric Validation Embedding (AVE) bias.

    This splitter uses a Genetic Algorithm to construct train and validation splits
    that have matched active/inactive neighbor distance distributions, avoiding
    analogue bias and decoy bias.

    Based on the AVE algorithm described in:
    Wallach, Izhar, and Abraham Heifets. "Most ligand-based benchmarks measure overfitting rather than accuracy."
    Journal of chemical information and modeling 58.3 (2018): 516-525.
    (arxiv:1706.06619)
    """

    def __init__(self,
                 metric: str = 'euclidean',
                 max_iter: int = 100,
                 pop_size: int = 100,
                 next_gen_size: int = 20,
                 mutation_rate: float = 0.2,
                 early_stopping_obj: float = 0.01):
        """
        Parameters
        ----------
        metric : str, optional (default 'euclidean')
            Distance metric to compute between features (e.g., 'euclidean', 'jaccard').
            If 'jaccard', features should be binary fingerprints.
        max_iter : int, optional (default 100)
            Maximum number of genetic algorithm iterations.
        pop_size : int, optional (default 100)
            Number of splits in the population per generation.
        next_gen_size : int, optional (default 20)
            Number of top splits to keep for breeding the next generation.
        mutation_rate : float, optional (default 0.2)
            Probability of swapping samples during breeding.
        early_stopping_obj : float, optional (default 0.01)
            Objective value below which the algorithm stops early.
        """
        super().__init__()
        self.metric = metric
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.next_gen_size = next_gen_size
        self.mutation_rate = mutation_rate
        self.early_stopping_obj = early_stopping_obj

    def split(self,
              dataset: Dataset,
              frac_train: float = 0.8,
              frac_valid: float = 0.2,
              frac_test: float = 0.0,
              seed: Optional[int] = None,
              log_every_n: Optional[int] = None,
              task_index: int = 0) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits dataset into train/valid/test sets minimizing AVE bias.

        Parameters
        ----------
        dataset : Dataset
            Dataset to split.
        frac_train : float, optional (default 0.8)
            Fraction of data to put in training set.
        frac_valid : float, optional (default 0.2)
            Fraction of data to put in validation set.
        frac_test : float, optional (default 0.0)
            Fraction of data to put in test set. (Typically 0 for AVE).
        seed : int, optional (default None)
            Random seed.
        log_every_n : int, optional (default None)
            Logging interval.
        task_index: int, optional (default 0)
            Index of the task to use for active/inactive labels.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            Indices for train, valid, and test sets.
        """
        from scipy.spatial.distance import cdist

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Only considering binary classification datasets where labels are 0 or 1
        y = dataset.y[:, task_index]
        if not np.array_equal(np.unique(y), [0, 1]) and not np.array_equal(np.unique(y), [0.0, 1.0]):
            raise ValueError("AVESplitter requires binary labels (0 and 1) for the specified task.")

        active_indices = np.where(y == 1)[0]
        inactive_indices = np.where(y == 0)[0]

        X_actives = dataset.X[active_indices]
        X_inactives = dataset.X[inactive_indices]

        # Calculate full pairwise distance matrices
        logger.info("Computing distance matrices...")
        aa_D = cdist(X_actives, X_actives, metric=self.metric)
        ii_D = cdist(X_inactives, X_inactives, metric=self.metric)
        ai_D = cdist(X_actives, X_inactives, metric=self.metric)
        ia_D = ai_D.T

        split_actives = int(len(active_indices) * frac_train / (frac_train + frac_valid + frac_test))
        split_inactives = int(len(inactive_indices) * frac_train / (frac_train + frac_valid + frac_test))

        # Initial random population
        pop = []
        act_idx_list = list(range(len(active_indices)))
        inact_idx_list = list(range(len(inactive_indices)))

        while len(pop) < self.pop_size:
            random.shuffle(act_idx_list)
            random.shuffle(inact_idx_list)
            pop.append((act_idx_list[:split_actives], inact_idx_list[:split_inactives],
                        act_idx_list[split_actives:], inact_idx_list[split_inactives:]))

        min_obj = float('inf')
        final_pop = None

        for iter_count in range(self.max_iter):
            objs = []
            for cvSet in pop:
                vaI, viI, taI, tiI = cvSet

                # slices of distance matrices
                # Here, "taI" is train actives, "vaI" is validation actives
                aTest_aTrain_D = aa_D[np.ix_(vaI, taI)]
                aTest_iTrain_D = ai_D[np.ix_(vaI, tiI)]
                iTest_aTrain_D = ia_D[np.ix_(viI, taI)]
                iTest_iTrain_D = ii_D[np.ix_(viI, tiI)]

                aTest_aTrain_S = np.mean([np.mean(np.any(aTest_aTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
                aTest_iTrain_S = np.mean([np.mean(np.any(aTest_iTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
                iTest_iTrain_S = np.mean([np.mean(np.any(iTest_iTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])
                iTest_aTrain_S = np.mean([np.mean(np.any(iTest_aTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 50)])

                # Objective minimizes bias
                obj = abs((aTest_aTrain_S - aTest_iTrain_S) + (iTest_iTrain_S - iTest_aTrain_S))
                objs.append(obj)

            # Sort population by objective
            combined_pop = sorted(zip(objs, pop), key=lambda x: x[0])

            # Select top next_gen_size
            new_pop = combined_pop[:self.next_gen_size]
            best_obj, best_split = new_pop[0]

            if log_every_n is not None and iter_count % log_every_n == 0:
                logger.info(f"Iteration {iter_count}, Best AVE Objective: {best_obj}")

            if best_obj < min_obj:
                min_obj = best_obj
                final_pop = best_split

            if min_obj < self.early_stopping_obj:
                break

            # Breed
            pop = []
            while len(pop) < self.pop_size:
                pair = random.sample(new_pop, 2)

                newActiveIndicesV = list(set(pair[0][1][0] + pair[1][1][0]))
                newInactiveIndicesV = list(set(pair[0][1][1] + pair[1][1][1]))
                newActiveIndicesT = list(set(pair[0][1][2] + pair[1][1][2]))
                newInactiveIndicesT = list(set(pair[0][1][3] + pair[1][1][3]))

                # Ensure no overlap
                for val in np.intersect1d(newActiveIndicesV, newActiveIndicesT):
                    if random.random() < 0.5:
                        newActiveIndicesV.remove(val)
                    else:
                        newActiveIndicesT.remove(val)

                for val in np.intersect1d(newInactiveIndicesV, newInactiveIndicesT):
                    if random.random() < 0.5:
                        newInactiveIndicesV.remove(val)
                    else:
                        newInactiveIndicesT.remove(val)

                # Mutate sizes slightly
                avSize = len(pair[0][1][0])
                ivSize = len(pair[0][1][1])
                atSize = len(pair[0][1][2])
                itSize = len(pair[0][1][3])

                if random.random() < self.mutation_rate:
                    avSize += random.choice([-1, 1])
                if random.random() < self.mutation_rate:
                    ivSize += random.choice([-1, 1])
                if random.random() < self.mutation_rate:
                    atSize += random.choice([-1, 1])
                if random.random() < self.mutation_rate:
                    itSize += random.choice([-1, 1])

                avSize = max(1, min(avSize, len(newActiveIndicesV)))
                ivSize = max(1, min(ivSize, len(newInactiveIndicesV)))
                atSize = max(1, min(atSize, len(newActiveIndicesT)))
                itSize = max(1, min(itSize, len(newInactiveIndicesT)))

                avSamp = random.sample(newActiveIndicesV, avSize)
                ivSamp = random.sample(newInactiveIndicesV, ivSize)
                atSamp = random.sample(newActiveIndicesT, atSize)
                itSamp = random.sample(newInactiveIndicesT, itSize)

                pop.append((avSamp, ivSamp, atSamp, itSamp))

        # Reconstruct actual indices in dataset
        vaI, viI, taI, tiI = final_pop

        train_indices = [active_indices[i] for i in taI] + [inactive_indices[i] for i in tiI]
        valid_indices = [active_indices[i] for i in vaI] + [inactive_indices[i] for i in viI]

        # If frac_test > 0, randomly split from validation
        test_indices = []
        if frac_test > 0:
            test_size = int(len(dataset) * frac_test)
            random.shuffle(valid_indices)
            test_indices = valid_indices[:test_size]
            valid_indices = valid_indices[test_size:]

        return train_indices, valid_indices, test_indices
