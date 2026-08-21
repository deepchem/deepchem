"""Evolution configuration and validation.

Provides the EvolutionConfig dataclass and validate_evolution_config().
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from deepchem.models.symbolic_regression.core.operators import Operator
from deepchem.models.symbolic_regression.genetic.mutation import validate_custom_mutation_weights

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary search."""

    # Population
    n_islands: int = 5
    population_size: int = 50
    n_cycles: int = 1000

    # Constraints
    max_size: int = 20
    max_depth: int = 8
    warmup_maxsize_by: Optional[int] = None
    warmup_start_size: int = 5

    # Selection
    tournament_size: int = 7
    tournament_pick_prob: float = 1.0
    crossover_prob: float = 0.1

    # Offspring
    offspring_per_cycle: int = 10
    max_attempts_per_offspring: int = 5
    avoid_duplicate_offspring: bool = True

    # Constant optimization
    optimize_prob: float = 0.15
    optimize_steps: int = 10
    optimize_lr: float = 0.01

    optimize_n_restarts: int = 0
    optimize_restart_noise: float = 0.1
    optimize_stagnation_boost: float = 0.0
    final_refine_topk: int = 5

    # Parsimony
    parsimony_coefficient: float = 0.001

    # Loss
    loss_type: str = "mse"
    loss_scale: str = "linear"
    huber_delta: float = 1.0
    loss_epsilon: float = 1e-12

    # Linear scaling
    linear_scaling: bool = False
    linear_scaling_min_improvement: float = 1e-12

    # Batching
    batching: Union[bool, str] = "auto"
    batch_size: Optional[int] = None
    batching_threshold: int = 1000
    batch_explore_prob: float = 0.1

    # Migration (HoF broadcast only)
    migration_interval: int = 50
    hof_migration_rate: float = 0.06
    topn_migrants: int = 12

    # Stagnation restart
    stagnation_patience: int = 8
    restart_elite_fraction: float = 0.0
    restart_hof_fraction: float = 0.5

    # Mutation weights (None = use defaults from mutation.py)
    mutation_weights: Optional[Dict[str, float]] = None

    # Operator constraints
    constraints: Optional[Dict[str, Union[int, Tuple[int, int]]]] = None
    nested_constraints: Optional[Dict[str, Dict[str, int]]] = None

    # Operators (None = use defaults)
    unary_ops: Optional[Dict[str, Operator]] = None
    binary_ops: Optional[Dict[str, Operator]] = None

    # Progress reporting
    verbose: bool = True
    print_every: int = 100

    # Reproducibility
    random_seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Constraint validation helpers
# ---------------------------------------------------------------------------


def _validate_constraints(
    constraints: Dict[str, Union[int, Tuple[int, int]]],) -> None:
    for op_name, limit in constraints.items():
        if not isinstance(op_name, str):
            raise ValueError(
                f"constraint key must be a string, got {op_name!r}")
        if isinstance(limit, int):
            if limit < -1:
                raise ValueError(
                    f"constraint for '{op_name}' must be >= -1, got {limit}")
        elif isinstance(limit, tuple) and len(limit) == 2:
            l, r = limit
            if not isinstance(l, int) or not isinstance(r, int):
                raise ValueError(
                    f"constraint tuple for '{op_name}' must contain ints, got {limit}"
                )
            if l < -1 or r < -1:
                raise ValueError(
                    f"constraint tuple for '{op_name}' must be >= -1, got {limit}"
                )
        else:
            raise ValueError(
                f"constraint for '{op_name}' must be int or (int, int), got {limit}"
            )


def _validate_nested_constraints(
    nested_constraints: Dict[str, Dict[str, int]],) -> None:
    for outer_op, inner_limits in nested_constraints.items():
        if not isinstance(outer_op, str):
            raise ValueError(
                f"nested constraint key must be a string, got {outer_op!r}")
        if not isinstance(inner_limits, dict):
            raise ValueError(
                f"nested constraint for '{outer_op}' must be a dict, got "
                f"{type(inner_limits)}")
        for inner_op, limit in inner_limits.items():
            if not isinstance(inner_op, str):
                raise ValueError(
                    f"nested inner key under '{outer_op}' must be a string, got "
                    f"{inner_op!r}")
            if not isinstance(limit, int) or limit < 0:
                raise ValueError(
                    f"nested limit for '{outer_op}->{inner_op}' must be "
                    f"a non-negative int, got {limit!r}")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_evolution_config(config: EvolutionConfig) -> None:
    """Validate search configuration fields."""
    if config.n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {config.n_cycles}")
    if config.population_size < 1:
        raise ValueError(
            f"population_size must be >= 1, got {config.population_size}")
    if config.n_islands < 1:
        raise ValueError(f"n_islands must be >= 1, got {config.n_islands}")
    if config.max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {config.max_size}")
    if config.max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {config.max_depth}")
    if config.warmup_maxsize_by is not None and config.warmup_maxsize_by < 1:
        raise ValueError(f"warmup_maxsize_by must be >= 1 when provided, got "
                         f"{config.warmup_maxsize_by}")
    if config.warmup_start_size < 1:
        raise ValueError(
            f"warmup_start_size must be >= 1, got {config.warmup_start_size}")
    if config.warmup_start_size > config.max_size:
        raise ValueError(f"warmup_start_size must be <= max_size, got "
                         f"{config.warmup_start_size} > {config.max_size}")
    if config.tournament_size < 1:
        raise ValueError(
            f"tournament_size must be >= 1, got {config.tournament_size}")
    if not (0.0 < config.tournament_pick_prob <= 1.0):
        raise ValueError(f"tournament_pick_prob must be in (0, 1], got "
                         f"{config.tournament_pick_prob}")
    if not (0.0 <= config.crossover_prob <= 1.0):
        raise ValueError(
            f"crossover_prob must be in [0, 1], got {config.crossover_prob}")
    if config.offspring_per_cycle < 1:
        raise ValueError(
            f"offspring_per_cycle must be >= 1, got {config.offspring_per_cycle}"
        )
    if config.max_attempts_per_offspring < 1:
        raise ValueError(f"max_attempts_per_offspring must be >= 1, got "
                         f"{config.max_attempts_per_offspring}")
    if not (0.0 <= config.optimize_prob <= 1.0):
        raise ValueError(
            f"optimize_prob must be in [0, 1], got {config.optimize_prob}")
    if config.optimize_steps < 1:
        raise ValueError(
            f"optimize_steps must be >= 1, got {config.optimize_steps}")
    if config.optimize_lr <= 0:
        raise ValueError(f"optimize_lr must be > 0, got {config.optimize_lr}")

    if config.optimize_n_restarts < 0:
        raise ValueError(
            f"optimize_n_restarts must be >= 0, got {config.optimize_n_restarts}"
        )
    if config.optimize_restart_noise < 0:
        raise ValueError(f"optimize_restart_noise must be >= 0, got "
                         f"{config.optimize_restart_noise}")
    if config.optimize_stagnation_boost < 0:
        raise ValueError(f"optimize_stagnation_boost must be >= 0, got "
                         f"{config.optimize_stagnation_boost}")
    if config.final_refine_topk < 0:
        raise ValueError(
            f"final_refine_topk must be >= 0, got {config.final_refine_topk}")
    if config.parsimony_coefficient < 0:
        raise ValueError(f"parsimony_coefficient must be >= 0, got "
                         f"{config.parsimony_coefficient}")
    if config.loss_type not in ("mse", "mae", "huber"):
        raise ValueError(
            f"loss_type must be one of ('mse', 'mae', 'huber'), got "
            f"{config.loss_type!r}")
    if config.loss_scale not in ("linear", "log"):
        raise ValueError(
            f"loss_scale must be 'linear' or 'log', got {config.loss_scale!r}")
    if config.huber_delta <= 0:
        raise ValueError(f"huber_delta must be > 0, got {config.huber_delta}")
    if config.loss_epsilon <= 0:
        raise ValueError(f"loss_epsilon must be > 0, got {config.loss_epsilon}")
    if config.batching != "auto" and not isinstance(config.batching, bool):
        raise ValueError(
            f"batching must be bool or 'auto', got {config.batching!r}")
    if config.batch_size is not None and config.batch_size < 1:
        raise ValueError(
            f"batch_size must be >= 1 when provided, got {config.batch_size}")
    if config.batching_threshold < 1:
        raise ValueError(
            f"batching_threshold must be >= 1, got {config.batching_threshold}")
    if not (0.0 <= config.batch_explore_prob <= 1.0):
        raise ValueError(f"batch_explore_prob must be in [0, 1], got "
                         f"{config.batch_explore_prob}")
    if config.migration_interval < 1:
        raise ValueError(
            f"migration_interval must be >= 1, got {config.migration_interval}")
    if not (0.0 <= config.hof_migration_rate <= 1.0):
        raise ValueError(f"hof_migration_rate must be in [0, 1], got "
                         f"{config.hof_migration_rate}")
    if config.topn_migrants < 1:
        raise ValueError(
            f"topn_migrants must be >= 1, got {config.topn_migrants}")
    if config.stagnation_patience < 1:
        raise ValueError(f"stagnation_patience must be >= 1, got "
                         f"{config.stagnation_patience}")
    if not (0.0 <= config.restart_elite_fraction <= 1.0):
        raise ValueError(f"restart_elite_fraction must be in [0, 1], got "
                         f"{config.restart_elite_fraction}")
    if not (0.0 <= config.restart_hof_fraction <= 1.0):
        raise ValueError(f"restart_hof_fraction must be in [0, 1], got "
                         f"{config.restart_hof_fraction}")
    if config.verbose and config.print_every < 1:
        raise ValueError(f"print_every must be >= 1 when verbose=True, got "
                         f"{config.print_every}")
    if config.mutation_weights is not None:
        validate_custom_mutation_weights(config.mutation_weights)
    if config.constraints is not None and not isinstance(
            config.constraints, dict):
        raise ValueError(f"constraints must be a dict or None, got "
                         f"{type(config.constraints)}")
    if (config.nested_constraints is not None and
            not isinstance(config.nested_constraints, dict)):
        raise ValueError(f"nested_constraints must be a dict or None, got "
                         f"{type(config.nested_constraints)}")
