"""Island model evolution for symbolic regression.

Orchestrates the full search: initializes islands of random
expression trees, runs mutation-evaluation cycles, migrates
HoF entries across islands, and returns the Pareto front.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from deepchem.models.symbolic_regression.core.node import (
    Node,
    VariableNode,
    ConstantNode,
    UnaryOpNode,
    BinaryOpNode,
)
from deepchem.models.symbolic_regression.core.operators import (
    Operator,
    DEFAULT_UNARY_OPS,
    DEFAULT_BINARY_OPS,
)
from deepchem.models.symbolic_regression.evaluation.evaluate import (
    compute_complexity,
    evaluate_tree,
)
from deepchem.models.symbolic_regression.evolution.objective import (
    _objective_loss,
    fit_linear_scaling,
    score_tree,
)
from deepchem.models.symbolic_regression.optimization.optimize import optimize_constants
from deepchem.models.symbolic_regression.simplification.simplify import simplify
from deepchem.models.symbolic_regression.genetic.selection import tournament_select
from deepchem.models.symbolic_regression.genetic.crossover import crossover
from deepchem.models.symbolic_regression.genetic.mutation import mutate_with_info
from deepchem.models.symbolic_regression.evolution.hall_of_fame import HallOfFame
from deepchem.models.symbolic_regression.evolution.config import (
    EvolutionConfig,
    validate_evolution_config,
    _validate_constraints,
    _validate_nested_constraints,
)
from deepchem.models.symbolic_regression.evolution.population import (
    _init_island,
    _migrate_from_hof,
    _restart_stagnant_islands,
    _final_refine_hof_entries,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_operator(node: Node, op_name: str) -> int:
    """Count occurrences of an operator in a subtree."""
    if isinstance(node, (VariableNode, ConstantNode)):
        return 0
    count = 1 if getattr(node, "op_name", None) == op_name else 0
    if hasattr(node, "child"):
        count += _count_operator(node.child, op_name)
    if hasattr(node, "left"):
        count += _count_operator(node.left, op_name)
    if hasattr(node, "right"):
        count += _count_operator(node.right, op_name)
    return count


def _tree_signature(node: Node) -> str:
    """Canonical signature for exact tree structure + constant values."""
    if isinstance(node, VariableNode):
        return f"V({node.index})"
    if isinstance(node, ConstantNode):
        return f"C({float(node.value.item()):.17g})"
    if isinstance(node, UnaryOpNode):
        return f"U({node.op_name},{_tree_signature(node.child)})"
    if isinstance(node, BinaryOpNode):
        return (f"B({node.op_name},{_tree_signature(node.left)},"
                f"{_tree_signature(node.right)})")
    return f"X({type(node).__name__})"


def _current_max_size(config: EvolutionConfig, cycle: int) -> int:
    """Return active max-size under optional warmup schedule."""
    if config.warmup_maxsize_by is None:
        return config.max_size
    grown = config.warmup_start_size + (cycle // config.warmup_maxsize_by)
    return min(config.max_size, grown)


def _resolve_batching(config: EvolutionConfig,
                      n_samples: int) -> Tuple[bool, int]:
    """Resolve batching mode and effective batch size."""
    mode = config.batching
    if mode == "auto":
        enabled = n_samples > config.batching_threshold
    elif isinstance(mode, bool):
        enabled = mode
    else:
        raise ValueError(f"batching must be bool or 'auto', got {mode!r}")

    if not enabled:
        return False, n_samples

    if config.batch_size is None:
        # 10% of dataset, floored at 128 to avoid tiny batches
        size = min(n_samples, max(128, n_samples // 10))
    else:
        size = config.batch_size

    size = max(1, min(n_samples, int(size)))
    return size < n_samples, size


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------


def _respects_constraints(
    node: Node,
    operators: Dict[str, Operator],
    constraints: Optional[Dict[str, Union[int, Tuple[int, int]]]] = None,
    nested_constraints: Optional[Dict[str, Dict[str, int]]] = None,
) -> bool:
    """Check whether an expression satisfies operator-side constraints."""
    if constraints is None and nested_constraints is None:
        return True
    if isinstance(node, (VariableNode, ConstantNode)):
        return True

    if hasattr(node, "child"):
        if not _respects_constraints(node.child, operators, constraints,
                                     nested_constraints):
            return False
    if hasattr(node, "left"):
        if not _respects_constraints(node.left, operators, constraints,
                                     nested_constraints):
            return False
    if hasattr(node, "right"):
        if not _respects_constraints(node.right, operators, constraints,
                                     nested_constraints):
            return False

    op_name = getattr(node, "op_name", None)
    if op_name is None:
        return True

    if constraints and op_name in constraints:
        limit = constraints[op_name]
        if hasattr(node, "left") and hasattr(node, "right"):
            if isinstance(limit, int):
                left_limit = right_limit = limit
            else:
                left_limit, right_limit = limit
            left_cplx = compute_complexity(node.left, operators)
            right_cplx = compute_complexity(node.right, operators)
            if left_limit >= 0 and left_cplx > left_limit:
                return False
            if right_limit >= 0 and right_cplx > right_limit:
                return False
        else:
            operand_limit = limit if isinstance(limit, int) else limit[0]
            if hasattr(node, "child"):
                child_cplx = compute_complexity(node.child, operators)
                if operand_limit >= 0 and child_cplx > operand_limit:
                    return False

    if nested_constraints and op_name in nested_constraints:
        inner_limits = nested_constraints[op_name]
        children = []
        if hasattr(node, "child"):
            children.append(node.child)
        if hasattr(node, "left"):
            children.append(node.left)
        if hasattr(node, "right"):
            children.append(node.right)
        for inner_op, limit in inner_limits.items():
            count = sum(_count_operator(child, inner_op) for child in children)
            if count > limit:
                return False

    return True


# ---------------------------------------------------------------------------
# Linear scaling & optimization
# ---------------------------------------------------------------------------


def _maybe_apply_linear_scaling(
    node: Node,
    X: Tensor,
    y: Tensor,
    all_ops: Dict[str, Operator],
    max_size_limit: int,
    max_depth: int,
    constraints=None,
    nested_constraints=None,
    enabled: bool = False,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    loss_epsilon: float = 1e-12,
    min_improvement: float = 1e-12,
) -> Node:
    """Optionally wrap node with affine output scaling when it improves loss."""
    if not enabled:
        return node
    if "mul" not in all_ops or "add" not in all_ops:
        return node

    with torch.no_grad():
        preds = evaluate_tree(node, X, all_ops)
        preds = torch.nan_to_num(preds, nan=1e6, posinf=1e6, neginf=-1e6)
        y_flat = y.flatten()
        base_loss = _objective_loss(preds,
                                    y_flat,
                                    loss_type=loss_type,
                                    huber_delta=huber_delta).item()
        scaled_preds, scale_a, shift_b = fit_linear_scaling(preds,
                                                            y_flat,
                                                            eps=loss_epsilon)
        scaled_loss = _objective_loss(scaled_preds,
                                      y_flat,
                                      loss_type=loss_type,
                                      huber_delta=huber_delta).item()

    if (not math.isfinite(base_loss) or not math.isfinite(scaled_loss) or
            scaled_loss + min_improvement >= base_loss):
        return node
    if abs(scale_a - 1.0) <= 1e-12 and abs(shift_b) <= 1e-12:
        return node

    scaled_node = BinaryOpNode(
        "add",
        BinaryOpNode("mul", ConstantNode(scale_a), node.copy()),
        ConstantNode(shift_b),
    )
    scaled_node = simplify(scaled_node)
    if (scaled_node.size() > max_size_limit or
            scaled_node.depth() > max_depth or not _respects_constraints(
                scaled_node, all_ops, constraints, nested_constraints)):
        return node
    return scaled_node


def _maybe_optimize_candidate(
    candidate: Node,
    do_optimize: bool,
    X: Tensor,
    y: Tensor,
    all_ops: Dict[str, Operator],
    config: EvolutionConfig,
    max_size_limit: int,
    constraints=None,
    nested_constraints=None,
) -> Node:
    """Optimize constants and apply linear scaling if configured."""
    result = candidate
    if do_optimize:
        optimized = optimize_constants(
            candidate.copy(),
            X,
            y,
            all_ops,
            steps=config.optimize_steps,
            lr=config.optimize_lr,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            n_restarts=config.optimize_n_restarts,
            restart_noise=config.optimize_restart_noise,
        )
        optimized = simplify(optimized)
        if (optimized.size() <= max_size_limit and
                optimized.depth() <= config.max_depth and _respects_constraints(
                    optimized, all_ops, constraints, nested_constraints)):
            result = optimized

    result = _maybe_apply_linear_scaling(
        result,
        X,
        y,
        all_ops,
        max_size_limit=max_size_limit,
        max_depth=config.max_depth,
        constraints=constraints,
        nested_constraints=nested_constraints,
        enabled=config.linear_scaling,
        loss_type=config.loss_type,
        huber_delta=config.huber_delta,
        loss_epsilon=config.loss_epsilon,
        min_improvement=config.linear_scaling_min_improvement,
    )
    return result


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------


def run_evolution(X, y, config: Optional[EvolutionConfig] = None) -> HallOfFame:
    """Run the full symbolic regression search.

    Parameters
    ----------
    X : array-like or Tensor
        Input data of shape (n_samples, n_features).
    y : array-like or Tensor
        Target values of shape (n_samples,).
    config : EvolutionConfig, optional
        Search configuration. Uses defaults if None.

    Returns
    -------
    HallOfFame
        Pareto front of best expressions at each complexity level.
    """
    if config is None:
        config = EvolutionConfig()

    rng = random.Random(config.random_seed)
    validate_evolution_config(config)

    # Convert to tensors if needed
    if not isinstance(X, Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")
    y = y.flatten()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")
    n_features = X.shape[1]
    if n_features < 1:
        raise ValueError(f"X must contain at least 1 feature, got {n_features}")

    use_batching, effective_batch_size = _resolve_batching(config, X.shape[0])

    unary_ops = (config.unary_ops
                 if config.unary_ops is not None else DEFAULT_UNARY_OPS)
    binary_ops = (config.binary_ops
                  if config.binary_ops is not None else DEFAULT_BINARY_OPS)
    if not unary_ops and not binary_ops:
        raise ValueError(
            "At least one unary or binary operator must be enabled")
    for name, op in unary_ops.items():
        if op.arity != 1:
            raise ValueError(
                f"unary operator '{name}' has arity {op.arity}, expected 1")
    for name, op in binary_ops.items():
        if op.arity != 2:
            raise ValueError(
                f"binary operator '{name}' has arity {op.arity}, expected 2")

    all_ops = {**unary_ops, **binary_ops}

    if config.constraints is not None:
        _validate_constraints(config.constraints)
        unknown = [n for n in config.constraints if n not in all_ops]
        if unknown:
            raise ValueError(
                f"constraints reference unknown operators: {unknown}")
    if config.nested_constraints is not None:
        _validate_nested_constraints(config.nested_constraints)
        unknown_outer = [
            n for n in config.nested_constraints if n not in all_ops
        ]
        if unknown_outer:
            raise ValueError(
                f"nested_constraints reference unknown outer operators: "
                f"{unknown_outer}")
        unknown_inner = []
        for outer, inner in config.nested_constraints.items():
            for name in inner:
                if name not in all_ops:
                    unknown_inner.append((outer, name))
        if unknown_inner:
            raise ValueError(
                f"nested_constraints reference unknown inner operators: "
                f"{unknown_inner}")

    hof = HallOfFame()

    # --- Initialize islands ---
    islands: List[List[Node]] = []
    island_fitnesses: List[List[float]] = []
    island_losses: List[List[float]] = []
    island_mses: List[List[float]] = []
    island_complexities: List[List[int]] = []

    for _ in range(config.n_islands):
        init_max_size = _current_max_size(config, cycle=0)
        pop, fits, losses, mses, complexities = _init_island(
            unary_ops,
            binary_ops,
            n_features,
            config.population_size,
            X,
            y,
            all_ops,
            config.parsimony_coefficient,
            hof,
            init_max_size,
            config.max_depth,
            rng,
            config.constraints,
            config.nested_constraints,
            loss_type=config.loss_type,
            loss_scale=config.loss_scale,
            huber_delta=config.huber_delta,
            loss_epsilon=config.loss_epsilon,
            linear_scaling=config.linear_scaling,
            linear_scaling_min_improvement=config.
            linear_scaling_min_improvement,
        )
        islands.append(pop)
        island_fitnesses.append(fits)
        island_losses.append(losses)
        island_mses.append(mses)
        island_complexities.append(complexities)

    island_best_losses = [min(ls) for ls in island_losses]
    island_stagnation = [0 for _ in range(config.n_islands)]

    if config.verbose:
        best = hof.get_best()
        if best:
            print(f"Initialized {config.n_islands} islands × "
                  f"{config.population_size} | Best MSE: {best.mse:.6f}")
        if use_batching:
            print(
                f"Batch prefilter enabled | batch_size={effective_batch_size} "
                f"(n_samples={X.shape[0]})")

    score_kw = dict(loss_type=config.loss_type,
                    loss_scale=config.loss_scale,
                    huber_delta=config.huber_delta,
                    loss_epsilon=config.loss_epsilon)

    # --- Main evolution loop ---
    for cycle in range(config.n_cycles):
        active_max_size = _current_max_size(config, cycle)

        for island_idx in range(config.n_islands):
            pop = islands[island_idx]
            fits = island_fitnesses[island_idx]
            losses = island_losses[island_idx]
            mses = island_mses[island_idx]
            complexities = island_complexities[island_idx]

            signature_set = set()
            if config.avoid_duplicate_offspring:
                signature_set = {_tree_signature(ind) for ind in pop}

            # Batch selection
            if use_batching and effective_batch_size < X.shape[0]:
                batch_idx = rng.sample(range(X.shape[0]), effective_batch_size)
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
            else:
                X_batch = X
                y_batch = y

            stagnation_ratio = min(
                1.0,
                max(
                    0.0, island_stagnation[island_idx] /
                    float(config.stagnation_patience)))
            optimize_prob = min(
                1.0,
                config.optimize_prob *
                (1.0 + config.optimize_stagnation_boost * stagnation_ratio),
            )
            opt_kwargs = {
                "X": X,
                "y": y,
                "operators": all_ops,
                "steps": config.optimize_steps,
                "lr": config.optimize_lr,
                "loss_type": config.loss_type,
                "huber_delta": config.huber_delta,
                "n_restarts": config.optimize_n_restarts,
                "restart_noise": config.optimize_restart_noise,
            }

            for _ in range(config.offspring_per_cycle):
                child = None
                child_mutation_name: Optional[str] = None

                for _ in range(config.max_attempts_per_offspring):
                    parent = tournament_select(
                        pop,
                        fits,
                        tournament_size=config.tournament_size,
                        pick_prob=config.tournament_pick_prob,
                        rng=rng)

                    if rng.random() < config.crossover_prob:
                        parent2 = tournament_select(
                            pop,
                            fits,
                            tournament_size=config.tournament_size,
                            pick_prob=config.tournament_pick_prob,
                            rng=rng)
                        proposal = crossover(parent, parent2, rng=rng)
                        proposal_mut_name: Optional[str] = None

                    else:
                        proposal = parent.copy()
                        proposal, proposal_mut_name = mutate_with_info(
                            proposal,
                            unary_ops,
                            binary_ops,
                            n_features,
                            max_depth=config.max_depth,
                            weights=config.mutation_weights,
                            optimize_fn=optimize_constants,
                            optimize_kwargs=opt_kwargs,
                            rng=rng,
                        )

                    if (proposal.size() > active_max_size or
                            proposal.depth() > config.max_depth):
                        continue

                    proposal = simplify(proposal)
                    if not _respects_constraints(proposal, all_ops,
                                                 config.constraints,
                                                 config.nested_constraints):
                        continue

                    child = proposal
                    child_mutation_name = proposal_mut_name
                    break

                if child is None:
                    continue

                child = _maybe_optimize_candidate(
                    child,
                    do_optimize=(rng.random() < optimize_prob and
                                 child_mutation_name != "optimize_constants"),
                    X=X,
                    y=y,
                    all_ops=all_ops,
                    config=config,
                    max_size_limit=active_max_size,
                    constraints=config.constraints,
                    nested_constraints=config.nested_constraints,
                )

                child_signature = ""
                if config.avoid_duplicate_offspring:
                    child_signature = _tree_signature(child)
                    if child_signature in signature_set:
                        continue

                # Find worst individual to replace
                replace_idx = max(range(len(fits)), key=lambda i: fits[i])

                # Batch prefilter
                if use_batching and effective_batch_size < X.shape[0]:
                    _, _, batch_fit, _ = score_tree(
                        child, X_batch, y_batch, all_ops,
                        config.parsimony_coefficient, **score_kw)
                    _, _, inc_batch_fit, _ = score_tree(
                        pop[replace_idx], X_batch, y_batch, all_ops,
                        config.parsimony_coefficient, **score_kw)
                    if (batch_fit > inc_batch_fit + 1e-8 and
                            rng.random() > config.batch_explore_prob):
                        continue

                # Full-data evaluation
                mse, loss, fitness, cplx = score_tree(
                    child, X, y, all_ops, config.parsimony_coefficient,
                    **score_kw)

                # Replace if better
                if fitness <= fits[replace_idx]:
                    if config.avoid_duplicate_offspring:
                        old_sig = _tree_signature(pop[replace_idx])
                        signature_set.discard(old_sig)
                        signature_set.add(child_signature)
                    pop[replace_idx] = child
                    fits[replace_idx] = fitness
                    losses[replace_idx] = loss
                    mses[replace_idx] = mse
                    complexities[replace_idx] = cplx

                # Update HoF (regardless of population replacement)
                hof.update(child, mse, fitness, complexity=cplx)

            # Track island-level stagnation
            current_best_loss = min(losses)
            if current_best_loss + 1e-12 < island_best_losses[island_idx]:
                island_best_losses[island_idx] = current_best_loss
                island_stagnation[island_idx] = 0
            else:
                island_stagnation[island_idx] += 1

        # --- Migration (HoF broadcast only) ---
        if cycle > 0 and cycle % config.migration_interval == 0:
            _migrate_from_hof(
                hof,
                islands,
                island_fitnesses,
                island_losses,
                island_mses,
                island_complexities,
                X,
                y,
                all_ops,
                config,
            )

        # --- Stagnation restart ---
        _restart_stagnant_islands(
            hof,
            islands,
            island_fitnesses,
            island_losses,
            island_mses,
            island_complexities,
            island_best_losses,
            island_stagnation,
            config,
            unary_ops,
            binary_ops,
            n_features,
            X,
            y,
            all_ops,
            active_max_size,
            config.parsimony_coefficient,
            rng,
        )

        # --- Progress reporting ---
        if config.verbose and (cycle + 1) % config.print_every == 0:
            best = hof.get_best()
            if best:
                print(f"Cycle {cycle + 1:5d}/{config.n_cycles} | "
                      f"Best MSE: {best.mse:.6f} | {best.node}")

    # --- Final refinement ---
    refined = _final_refine_hof_entries(
        hof,
        X,
        y,
        all_ops,
        config,
        config.parsimony_coefficient,
    )
    if config.verbose and config.final_refine_topk > 0:
        print(f"Final HoF refinement: top_k={config.final_refine_topk} | "
              f"improvements={refined}")

    if config.verbose:
        print("\n" + str(hof))

    return hof
