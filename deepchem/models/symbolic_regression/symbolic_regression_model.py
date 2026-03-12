"""DeepChem wrapper for Symbolic Regression.

Integrates the symbolic regression engine with DeepChem's Model API,
enabling use with DeepChem Datasets, metrics, and evaluation pipelines.

Example
-------
>>> import deepchem as dc
>>> import numpy as np
>>> X = np.random.randn(100, 2).astype(np.float32)
>>> y = (2 * X[:, 0] + X[:, 1] ** 2).astype(np.float32)
>>> dataset = dc.data.NumpyDataset(X, y)
>>> model = dc.models.SymbolicRegressionModel(n_cycles=500)
>>> model.fit(dataset)
>>> predictions = model.predict(dataset)
>>> print(model.get_equation())
"""

import logging
import os
from typing import Dict, List, Literal, Optional, Union, cast

import numpy as np
import torch

from deepchem.data import Dataset
from deepchem.models import Model
from deepchem.utils.data_utils import load_from_disk, save_to_disk

from deepchem.models.symbolic_regression.core.node import (
    Node,
    VariableNode,
    UnaryOpNode,
    BinaryOpNode,
)
from deepchem.models.symbolic_regression.core.operators import (
    DEFAULT_BINARY_OPS,
    DEFAULT_UNARY_OPS,
)
from deepchem.models.symbolic_regression.evaluation.evaluate import evaluate_tree
from deepchem.models.symbolic_regression.evolution.evolve import (
    EvolutionConfig,
    run_evolution,
    validate_evolution_config,
)
from deepchem.models.symbolic_regression.evolution.hall_of_fame import (
    HallOfFame,
    HoFEntry,
)
from deepchem.models.symbolic_regression.evolution.model_selection import (
    SelectionMethod,
    evaluate_hof_on_validation,
    select_entry,
)
from deepchem.models.symbolic_regression.optimization.optimize import optimize_constants

logger = logging.getLogger(__name__)


class SymbolicRegressionModel(Model):
    """Symbolic regression model for DeepChem.

    Discovers interpretable mathematical expressions that fit data
    using evolutionary search with island model, constant optimization,
    and algebraic simplification.

    Unlike neural networks, this model produces human-readable equations,
    making it ideal for scientific discovery and explainable AI.

    This model can be used with DeepChem's standard ``evaluate()`` method
    and ``dc.metrics.Metric`` objects for consistent model comparison.

    Examples
    --------
    Fit a model and retrieve the best equation:

    >>> import deepchem as dc
    >>> import numpy as np
    >>> X = np.random.randn(100, 1).astype(np.float32)
    >>> y = (2 * X[:, 0] + 1).astype(np.float32)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> model = dc.models.SymbolicRegressionModel(
    ...     n_cycles=200, verbose=False)  # doctest: +SKIP
    >>> model.fit(dataset)  # doctest: +SKIP
    >>> model.get_equation()  # doctest: +SKIP
    'add(mul(2.0, x0), 1.0)'

    Evaluate with DeepChem metrics:

    >>> from deepchem.metrics import Metric, mean_squared_error
    >>> score = model.evaluate(
    ...     dataset,
    ...     [Metric(mean_squared_error)])  # doctest: +SKIP

    Notes
    -----
    All computation is done in memory using PyTorch tensors.
    For large datasets, consider enabling ``batching="auto"``.
    Sample weights (``dataset.w``) are not currently supported and
    will be ignored during training.
    """

    def __init__(self,
                 n_islands: int = 5,
                 population_size: int = 50,
                 n_cycles: int = 1000,
                 max_size: int = 20,
                 max_depth: int = 8,
                 warmup_maxsize_by: Optional[int] = None,
                 warmup_start_size: int = 5,
                 tournament_size: int = 7,
                 tournament_pick_prob: float = 1.0,
                 crossover_prob: float = 0.1,
                 offspring_per_cycle: int = 10,
                 max_attempts_per_offspring: int = 5,
                 batching: Union[bool, Literal["auto"]] = "auto",
                 batch_size: Optional[int] = None,
                 batching_threshold: int = 1000,
                 batch_explore_prob: float = 0.1,
                 optimize_prob: float = 0.15,
                 optimize_steps: int = 10,
                 optimize_lr: float = 0.01,
                 optimize_n_restarts: int = 0,
                 optimize_restart_noise: float = 0.1,
                 optimize_stagnation_boost: float = 0.0,
                 final_refine_topk: int = 5,
                 parsimony_coefficient: float = 0.001,
                 loss_type: Literal["mse", "mae", "huber"] = "mse",
                 loss_scale: Literal["linear", "log"] = "linear",
                 huber_delta: float = 1.0,
                 loss_epsilon: float = 1e-12,
                 linear_scaling: bool = False,
                 linear_scaling_min_improvement: float = 1e-12,
                 migration_interval: int = 50,
                 hof_migration_rate: float = 0.06,
                 topn_migrants: int = 12,
                 stagnation_patience: int = 8,
                 restart_elite_fraction: float = 0.0,
                 restart_hof_fraction: float = 0.5,
                 random_seed: Optional[int] = None,
                 search_seeds: Optional[List[int]] = None,
                 mutation_weights: Optional[Dict[str, float]] = None,
                 avoid_duplicate_offspring: bool = True,
                 constraints: Optional[Dict] = None,
                 nested_constraints: Optional[Dict[str, Dict[str, int]]] = None,
                 unary_operator_names: Optional[List[str]] = None,
                 binary_operator_names: Optional[List[str]] = None,
                 validation_ratio: float = 0.0,
                 validation_seed: Optional[int] = None,
                 selection_method: Literal["train_mse", "accuracy", "score",
                                           "best"] = "train_mse",
                 best_loss_factor: float = 1.5,
                 refit_selected_constants: bool = True,
                 verbose: bool = True,
                 print_every: int = 100,
                 model_dir: Optional[str] = None):
        """
        Parameters
        ----------
        n_islands : int
            Number of independent populations. Default 5.
        population_size : int
            Individuals per island. Default 50.
        n_cycles : int
            Number of evolution cycles. Default 1000.
        max_size : int
            Maximum tree size (nodes). Default 20.
        max_depth : int
            Maximum tree depth. Default 8.
        warmup_maxsize_by : int, optional
            If set, starts search at `warmup_start_size` and increases the
            active max size by 1 every `warmup_maxsize_by` cycles.
        warmup_start_size : int
            Initial active max size when warmup is enabled. Default 5.
        tournament_size : int
            Tournament selection size. Default 7.
        tournament_pick_prob : float
            Probability of selecting the best member in each tournament.
            Values < 1.0 add diversity. Default 1.0.
        crossover_prob : float
            Probability of crossover vs mutation. Default 0.1.
        offspring_per_cycle : int
            Number of offspring generated per island per cycle. Default 10.
        max_attempts_per_offspring : int
            Number of attempts to generate a valid offspring before skipping
            that offspring slot. Default 5.
        batching : bool or "auto"
            Enable mini-batch prefiltering for candidate screening.
            "auto" enables when dataset size exceeds `batching_threshold`.
        batch_size : int, optional
            Batch size when batching is enabled. If None, a heuristic is used.
        batching_threshold : int
            Dataset-size threshold used by `batching="auto"`.
        batch_explore_prob : float
            Chance to bypass batch rejection and still run full evaluation.
        optimize_prob : float
            Probability of constant optimization per new expression.
        optimize_steps : int
            Gradient descent steps for constant optimization. Default 10.
        optimize_lr : float
            Learning rate for constant optimization. Default 0.01.
        optimize_n_restarts : int
            Number of random restarts for constant optimization.
        optimize_restart_noise : float
            Stddev of restart noise for constants.
        optimize_stagnation_boost : float
            Multiplier for increasing optimize probability as an island
            stagnates.
        final_refine_topk : int
            Number of top HoF equations to run through heavier constant
            optimization at end of evolution. Set 0 to disable.
        parsimony_coefficient : float
            Complexity penalty. Default 0.001.
        loss_type : {"mse", "mae", "huber"}
            Objective loss used during evolution fitness ranking.
        loss_scale : {"linear", "log"}
            Whether to use the raw objective loss or log(loss) in fitness.
        huber_delta : float
            Delta threshold used only when `loss_type="huber"`.
        loss_epsilon : float
            Small positive epsilon used for log-loss scaling stability.
        linear_scaling : bool
            If True, applies affine output correction `a*f(x)+b`.
        linear_scaling_min_improvement : float
            Minimum objective-loss decrease required to accept linear scaling.
        migration_interval : int
            Cycles between HoF migrations. Default 50.
        hof_migration_rate : float
            Fraction of each island replaced by HoF migrants. Default 0.06.
        topn_migrants : int
            Number of top HoF entries used as migration source. Default 12.
        stagnation_patience : int
            Cycles without improvement before an island restart. Default 8.
        restart_elite_fraction : float
            Fraction of restarted island preserved from its current best.
        restart_hof_fraction : float
            Fraction of restarted island seeded from HoF. Default 0.5.
        random_seed : int, optional
            Search RNG seed for reproducible evolution behavior.
        search_seeds : list[int], optional
            Run multiple independent searches with these seeds and merge
            the discovered HoF candidates. Improves reliability.
        mutation_weights : dict, optional
            Custom mutation type weights.
        avoid_duplicate_offspring : bool
            If True, skips offspring that are exact duplicates of existing
            island members. Default True.
        constraints : dict, optional
            Optional operator-side complexity constraints.
        nested_constraints : dict[str, dict[str, int]], optional
            Nested operator constraints, e.g. `{"sin": {"sin": 0}}`.
        unary_operator_names : list[str], optional
            Restrict unary operator set to these names.
        binary_operator_names : list[str], optional
            Restrict binary operator set to these names.
        validation_ratio : float
            Fraction of training data held out for final-model selection.
        validation_seed : int, optional
            Seed used for train/validation split permutation.
        selection_method : {"train_mse", "accuracy", "score", "best"}
            Final equation chooser. "train_mse" keeps legacy behavior.
        best_loss_factor : float
            Used only for `selection_method="best"`.
        refit_selected_constants : bool
            If True and validation-based selection is enabled, run a final
            constant refit on the full training data.
        verbose : bool
            Print progress during evolution. Default True.
        print_every : int
            Print interval in cycles. Default 100.
        model_dir : str, optional
            Directory to store model files.
        """
        if not (0.0 <= validation_ratio < 1.0):
            raise ValueError(
                f"validation_ratio must be in [0, 1), got {validation_ratio}")
        valid_methods = {"train_mse", "accuracy", "score", "best"}
        if selection_method not in valid_methods:
            raise ValueError(
                "selection_method must be one of "
                f"{sorted(valid_methods)}, got {selection_method!r}")
        if validation_ratio == 0.0 and selection_method != "train_mse":
            raise ValueError(
                "selection_method requires validation_ratio > 0. "
                "Use selection_method='train_mse' or set validation_ratio.")
        if best_loss_factor <= 0:
            raise ValueError(
                f"best_loss_factor must be > 0, got {best_loss_factor}")
        if not isinstance(refit_selected_constants, bool):
            raise ValueError("refit_selected_constants must be bool, got "
                             f"{type(refit_selected_constants)}")
        if search_seeds is not None:
            if len(search_seeds) == 0:
                raise ValueError(
                    "search_seeds must be non-empty when provided.")
            invalid = [s for s in search_seeds if not isinstance(s, int)]
            if invalid:
                raise ValueError(
                    f"search_seeds must contain only ints, got {invalid}")

        def _select_ops(
            selected_names: Optional[List[str]],
            available: Dict[str, object],
            label: str,
        ) -> Dict[str, object]:
            if selected_names is None:
                return dict(available)
            if len(selected_names) == 0:
                return {}
            unknown = [name for name in selected_names if name not in available]
            if unknown:
                raise ValueError(f"Unknown {label} operators: {unknown}. "
                                 f"Available: {sorted(available.keys())}")
            deduped_names = list(dict.fromkeys(selected_names))
            return {name: available[name] for name in deduped_names}

        selected_unary_ops = _select_ops(unary_operator_names,
                                         DEFAULT_UNARY_OPS, "unary")
        selected_binary_ops = _select_ops(binary_operator_names,
                                          DEFAULT_BINARY_OPS, "binary")
        if not selected_unary_ops and not selected_binary_ops:
            raise ValueError(
                "At least one unary or binary operator must be enabled.")

        self._validation_ratio = float(validation_ratio)
        self._validation_seed = validation_seed
        self._selection_method = selection_method
        self._best_loss_factor = float(best_loss_factor)
        self._refit_selected_constants = refit_selected_constants
        self._search_seeds = (list(dict.fromkeys(search_seeds))
                              if search_seeds is not None else None)

        # Build evolution config from parameters
        self._config = EvolutionConfig(
            n_islands=n_islands,
            population_size=population_size,
            n_cycles=n_cycles,
            max_size=max_size,
            max_depth=max_depth,
            warmup_maxsize_by=warmup_maxsize_by,
            warmup_start_size=warmup_start_size,
            tournament_size=tournament_size,
            tournament_pick_prob=tournament_pick_prob,
            crossover_prob=crossover_prob,
            offspring_per_cycle=offspring_per_cycle,
            max_attempts_per_offspring=max_attempts_per_offspring,
            avoid_duplicate_offspring=avoid_duplicate_offspring,
            optimize_prob=optimize_prob,
            optimize_steps=optimize_steps,
            optimize_lr=optimize_lr,
            optimize_n_restarts=optimize_n_restarts,
            optimize_restart_noise=optimize_restart_noise,
            optimize_stagnation_boost=optimize_stagnation_boost,
            final_refine_topk=final_refine_topk,
            parsimony_coefficient=parsimony_coefficient,
            loss_type=loss_type,
            loss_scale=loss_scale,
            huber_delta=huber_delta,
            loss_epsilon=loss_epsilon,
            linear_scaling=linear_scaling,
            linear_scaling_min_improvement=linear_scaling_min_improvement,
            batching=batching,
            batch_size=batch_size,
            batching_threshold=batching_threshold,
            batch_explore_prob=batch_explore_prob,
            migration_interval=migration_interval,
            hof_migration_rate=hof_migration_rate,
            topn_migrants=topn_migrants,
            stagnation_patience=stagnation_patience,
            restart_elite_fraction=restart_elite_fraction,
            restart_hof_fraction=restart_hof_fraction,
            random_seed=random_seed,
            mutation_weights=mutation_weights,
            constraints=constraints,
            nested_constraints=nested_constraints,
            unary_ops=selected_unary_ops,
            binary_ops=selected_binary_ops,
            verbose=verbose,
            print_every=print_every,
        )
        validate_evolution_config(self._config)

        # Results (populated after fit)
        self._hof: Optional[HallOfFame] = None
        self._best_entry: Optional[HoFEntry] = None
        self._n_features: Optional[int] = None
        self._operators = {**selected_unary_ops, **selected_binary_ops}

        # Initialize base Model.
        # model=None because SR uses evolutionary search, not a wrapped
        # sklearn / torch model object.
        super(SymbolicRegressionModel, self).__init__(model=None,
                                                      model_dir=model_dir)

    def fit(self, dataset: Dataset) -> None:
        """Fit the model by running evolutionary symbolic regression.

        Parameters
        ----------
        dataset : Dataset
            DeepChem Dataset with X (features) and y (targets).
            Sample weights (``dataset.w``) are not used.
        """
        X = np.asarray(dataset.X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")

        y = np.asarray(dataset.y, dtype=np.float32)
        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(
                    f"SymbolicRegressionModel currently supports only a single "
                    f"target. Got y with shape {y.shape}")
            y = y[:, 0]
        else:
            raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples, got {X.shape[0]} "
                f"and {y.shape[0]}")

        self._n_features = X.shape[1]

        # Warn if non-uniform sample weights are present (not supported).
        w = np.asarray(dataset.w).flatten()
        if w.size > 0 and not np.allclose(w, w[0]):
            logger.warning(
                "SymbolicRegressionModel does not support sample weights. "
                "Non-uniform weights in dataset.w will be ignored.")

        logger.info(f"Starting symbolic regression: "
                    f"{X.shape[0]} samples, {self._n_features} features")

        X_fit = X
        y_fit = y
        X_val: Optional[np.ndarray] = None
        y_val: Optional[np.ndarray] = None

        if self._validation_ratio > 0.0:
            n_samples = X.shape[0]
            if n_samples < 3:
                raise ValueError(
                    "validation_ratio > 0 requires at least 3 samples.")
            n_val = int(round(self._validation_ratio * n_samples))
            n_val = max(1, min(n_samples - 1, n_val))
            split_seed = (self._validation_seed if self._validation_seed
                          is not None else self._config.random_seed)
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n_samples)
            val_idx = perm[:n_val]
            fit_idx = perm[n_val:]
            X_fit = X[fit_idx]
            y_fit = y[fit_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

        if self._search_seeds is not None:
            seeds = self._search_seeds
        elif self._config.random_seed is not None:
            seeds = [self._config.random_seed]
        else:
            seeds = [0]

        if self._config.verbose and len(seeds) > 1:
            logger.info("Running %d seed restarts: %s", len(seeds), seeds)

        merged_hof = HallOfFame()
        for seed in seeds:
            run_config = EvolutionConfig(**self._config.__dict__)
            run_config.random_seed = seed
            # Avoid noisy duplicated logs when running multiple restarts.
            run_config.verbose = self._config.verbose and len(seeds) == 1
            hof = run_evolution(X_fit, y_fit, run_config)
            if self._config.verbose and len(seeds) > 1:
                best = hof.get_best()
                if best is not None:
                    logger.info("Seed %s best MSE=%.6g", seed, best.mse)
                else:
                    logger.info("Seed %s produced empty HoF", seed)
            for entry in hof.entries.values():
                merged_hof.update(
                    entry.node,
                    entry.mse,
                    entry.fitness,
                    complexity=entry.complexity,
                )

        self._hof = merged_hof
        self._best_entry = self._hof.get_best()

        if (self._validation_ratio > 0.0 and X_val is not None and
                y_val is not None and self._selection_method != "train_mse" and
                self._hof is not None):
            entries = list(self._hof.entries.values())
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            stats = evaluate_hof_on_validation(entries, X_val_tensor,
                                               y_val_tensor, self._operators)
            method = cast(SelectionMethod, self._selection_method)
            chosen = select_entry(stats,
                                  method=method,
                                  best_loss_factor=self._best_loss_factor)
            if chosen is not None:
                self._best_entry = chosen

        if (self._refit_selected_constants and self._validation_ratio > 0.0 and
                self._best_entry is not None):
            X_full_tensor = torch.tensor(X, dtype=torch.float32)
            y_full_tensor = torch.tensor(y, dtype=torch.float32)

            with torch.no_grad():
                baseline_preds = evaluate_tree(self._best_entry.node,
                                               X_full_tensor, self._operators)
                baseline_preds = torch.nan_to_num(baseline_preds,
                                                  nan=1e6,
                                                  posinf=1e6,
                                                  neginf=-1e6)
                baseline_mse = float(
                    torch.mean(
                        (baseline_preds - y_full_tensor.flatten())**2).item())

            refit_node = optimize_constants(
                self._best_entry.node.copy(),
                X_full_tensor,
                y_full_tensor,
                self._operators,
                steps=max(self._config.optimize_steps,
                          4 * self._config.optimize_steps),
                lr=self._config.optimize_lr,
                loss_type=self._config.loss_type,
                huber_delta=self._config.huber_delta,
                n_restarts=max(self._config.optimize_n_restarts, 2),
                restart_noise=self._config.optimize_restart_noise,
            )

            with torch.no_grad():
                refit_preds = evaluate_tree(refit_node, X_full_tensor,
                                            self._operators)
                refit_preds = torch.nan_to_num(refit_preds,
                                               nan=1e6,
                                               posinf=1e6,
                                               neginf=-1e6)
                refit_mse = float(
                    torch.mean(
                        (refit_preds - y_full_tensor.flatten())**2).item())

            if refit_mse + 1e-12 < baseline_mse:
                self._best_entry.node = refit_node
                self._best_entry.mse = refit_mse

        if self._best_entry and self._config.verbose:
            logger.info(f"Best expression: {self._best_entry.node} "
                        f"(MSE={self._best_entry.mse:.6f})")

    def predict_on_batch(self, X: np.typing.ArrayLike) -> np.ndarray:
        """Make predictions on a batch of data.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).
        """
        if self._best_entry is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        if self._n_features is None:
            raise RuntimeError(
                "Model is missing feature metadata. Refit or reload.")

        X_array = np.asarray(X, dtype=np.float32)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X_array.shape}")
        if X_array.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} input features, got {X_array.shape[1]}"
            )

        X_tensor = torch.tensor(X_array)

        with torch.no_grad():
            preds = evaluate_tree(self._best_entry.node, X_tensor,
                                  self._operators)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)

        return preds.numpy()

    # predict() is inherited from Model base class — it calls
    # predict_on_batch() and handles transformer undo automatically.

    def get_equation(self) -> str:
        """Return the best discovered equation as a string.

        Returns
        -------
        str
            Human-readable equation string.

        Raises
        ------
        RuntimeError
            If model has not been fit yet.
        """
        if self._best_entry is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return str(self._best_entry.node)

    def get_pareto_front(self) -> List[dict]:
        """Return the Pareto front of accuracy vs complexity.

        Each entry is a dict with 'complexity', 'mse', and 'equation'.

        Returns
        -------
        list[dict]
            Sorted by complexity (ascending).

        Raises
        ------
        RuntimeError
            If model has not been fit yet.
        """
        if self._hof is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        front = []
        for complexity, entry in self._hof.get_pareto_front():
            front.append({
                'complexity': complexity,
                'mse': entry.mse,
                'equation': str(entry.node),
            })
        return front

    def get_hall_of_fame(self) -> Optional[HallOfFame]:
        """Return the raw HallOfFame object.

        Returns
        -------
        HallOfFame or None
            The HallOfFame if model has been fit, else None.
        """
        return self._hof

    def get_task_type(self) -> str:
        """Return task type for DeepChem compatibility."""
        return "regression"

    def get_num_tasks(self) -> int:
        """Return number of tasks for DeepChem compatibility."""
        return 1

    def save(self) -> None:
        """Save model to disk.

        Saves the HallOfFame entries and config for later reloading.
        Uses ``deepchem.utils.data_utils.save_to_disk`` for consistency
        with other DeepChem models.
        """
        save_path = os.path.join(self.model_dir, "symbolic_regression.joblib")
        data = {
            'hof': self._hof,
            'best_entry': self._best_entry,
            'config': self._config,
            'n_features': self._n_features,
        }
        save_to_disk(data, save_path)
        logger.info(f"Model saved to {save_path}")

    def reload(self) -> None:
        """Load model from disk.

        Uses ``deepchem.utils.data_utils.load_from_disk`` for consistency
        with other DeepChem models.
        """
        save_path = os.path.join(self.model_dir, "symbolic_regression.joblib")
        data = load_from_disk(save_path)
        self._hof = data['hof']
        self._best_entry = data['best_entry']
        self._config = data['config']
        unary_ops = (self._config.unary_ops if self._config.unary_ops
                     is not None else DEFAULT_UNARY_OPS)
        binary_ops = (self._config.binary_ops if self._config.binary_ops
                      is not None else DEFAULT_BINARY_OPS)
        self._operators = {**unary_ops, **binary_ops}
        self._n_features = data.get('n_features')
        if self._n_features is None and self._best_entry is not None:
            self._n_features = self._infer_n_features(self._best_entry.node)
        logger.info(f"Model loaded from {save_path}")

    @staticmethod
    def _infer_n_features(node: Node) -> int:
        """Infer minimum required number of features from a tree."""
        if isinstance(node, VariableNode):
            return node.index + 1
        if isinstance(node, UnaryOpNode):
            return SymbolicRegressionModel._infer_n_features(node.child)
        if isinstance(node, BinaryOpNode):
            return max(
                SymbolicRegressionModel._infer_n_features(node.left),
                SymbolicRegressionModel._infer_n_features(node.right),
            )
        return 1
