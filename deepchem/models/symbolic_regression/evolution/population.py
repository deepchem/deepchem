"""Population management for island-model evolution.

Handles island initialization, HoF-based migration, stagnation restarts,
and final HoF refinement.
"""

import random

from deepchem.models.symbolic_regression.core.node import VariableNode
from deepchem.models.symbolic_regression.evolution.objective import score_tree
from deepchem.models.symbolic_regression.optimization.optimize import optimize_constants
from deepchem.models.symbolic_regression.simplification.simplify import simplify
from deepchem.models.symbolic_regression.genetic.mutation import (
    mutate_with_info,
    generate_random_tree,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_valid_random_tree(
    idx,
    unary_ops,
    binary_ops,
    n_features,
    max_depth,
    max_size,
    all_ops,
    constraints,
    nested_constraints,
    rng,
):
    """Generate a valid random tree with ramped half-and-half initialization.

    Cycles through grow/full methods and depth levels. Retries up to 10 times
    if the candidate violates constraints. Falls back to VariableNode(0).
    """
    from deepchem.models.symbolic_regression.evolution.evolve import _respects_constraints

    rand = rng if rng is not None else random
    init_depth_max = min(3, max_depth)
    init_depth_min = 1 if init_depth_max >= 1 else 0
    depth_span = max(1, init_depth_max - init_depth_min + 1)

    init_method = "grow" if idx % 2 == 0 else "full"
    init_depth = init_depth_min + ((idx // 2) % depth_span)

    for _ in range(10):
        candidate = generate_random_tree(unary_ops,
                                         binary_ops,
                                         n_features,
                                         max_depth=init_depth,
                                         method=init_method,
                                         rng=rand)
        candidate = simplify(candidate)
        if (candidate.size() <= max_size and
                candidate.depth() <= max_depth and _respects_constraints(
                    candidate, all_ops, constraints, nested_constraints)):
            return candidate
    return VariableNode(0)


# ---------------------------------------------------------------------------
# Island initialization
# ---------------------------------------------------------------------------


def _init_island(
    unary_ops,
    binary_ops,
    n_features,
    population_size,
    X,
    y,
    all_ops,
    parsimony_coefficient,
    hof,
    max_size,
    max_depth,
    rng=None,
    constraints=None,
    nested_constraints=None,
    loss_type="mse",
    loss_scale="linear",
    huber_delta=1.0,
    loss_epsilon=1e-12,
    linear_scaling=False,
    linear_scaling_min_improvement=1e-12,
):
    """Create one island with random trees, evaluated and scored."""
    from deepchem.models.symbolic_regression.evolution.evolve import (
        _maybe_apply_linear_scaling,)

    population, fitnesses, losses, mses, complexities = [], [], [], [], []
    rand = rng if rng is not None else random

    for idx in range(population_size):
        tree = _create_valid_random_tree(
            idx,
            unary_ops,
            binary_ops,
            n_features,
            max_depth,
            max_size,
            all_ops,
            constraints,
            nested_constraints,
            rand,
        )

        tree = _maybe_apply_linear_scaling(
            tree,
            X,
            y,
            all_ops,
            max_size_limit=max_size,
            max_depth=max_depth,
            constraints=constraints,
            nested_constraints=nested_constraints,
            enabled=linear_scaling,
            loss_type=loss_type,
            huber_delta=huber_delta,
            loss_epsilon=loss_epsilon,
            min_improvement=linear_scaling_min_improvement,
        )
        mse, loss, fitness, cplx = score_tree(
            tree,
            X,
            y,
            all_ops,
            parsimony_coefficient,
            loss_type=loss_type,
            loss_scale=loss_scale,
            huber_delta=huber_delta,
            loss_epsilon=loss_epsilon,
        )
        population.append(tree)
        fitnesses.append(fitness)
        losses.append(loss)
        mses.append(mse)
        complexities.append(cplx)
        hof.update(tree, mse, fitness, complexity=cplx)

    return population, fitnesses, losses, mses, complexities


# ---------------------------------------------------------------------------
# Migration (HoF broadcast only)
# ---------------------------------------------------------------------------


def _migrate_from_hof(hof, islands, island_fitnesses, island_losses,
                      island_mses, island_complexities, X, y, all_ops, config):
    """Inject top HoF members into each island, replacing worst."""
    hof_entries = sorted(hof.entries.values(), key=lambda e: e.mse)
    if not hof_entries:
        return
    topn = min(config.topn_migrants, len(hof_entries))
    if topn <= 0:
        return
    top_entries = hof_entries[:topn]

    kw = dict(loss_type=config.loss_type,
              loss_scale=config.loss_scale,
              huber_delta=config.huber_delta,
              loss_epsilon=config.loss_epsilon)

    for island_idx in range(len(islands)):
        pop = islands[island_idx]
        fits = island_fitnesses[island_idx]
        n_replace = max(1, int(len(pop) * config.hof_migration_rate))
        n_replace = min(n_replace, len(pop))
        if n_replace <= 0:
            continue

        worst_indices = sorted(range(len(pop)),
                               key=lambda i: fits[i],
                               reverse=True)
        for slot, target_idx in enumerate(worst_indices[:n_replace]):
            entry = top_entries[slot % len(top_entries)]
            migrant = entry.node.copy()
            mse, loss, fitness, cplx = score_tree(migrant, X, y, all_ops,
                                                  config.parsimony_coefficient,
                                                  **kw)
            pop[target_idx] = migrant
            fits[target_idx] = fitness
            island_losses[island_idx][target_idx] = loss
            island_mses[island_idx][target_idx] = mse
            island_complexities[island_idx][target_idx] = cplx


# ---------------------------------------------------------------------------
# Stagnation restart
# ---------------------------------------------------------------------------


def _restart_stagnant_islands(
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
    max_size_limit,
    parsimony_coefficient,
    rng=None,
):
    """Restart islands that have not improved for stagnation_patience cycles."""
    from deepchem.models.symbolic_regression.evolution.evolve import (
        _maybe_apply_linear_scaling,
        _maybe_optimize_candidate,
        _respects_constraints,
    )

    if config.stagnation_patience < 1:
        return

    rand = rng if rng is not None else random
    hof_entries = sorted(hof.entries.values(), key=lambda e: e.mse)
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
    kw = dict(loss_type=config.loss_type,
              loss_scale=config.loss_scale,
              huber_delta=config.huber_delta,
              loss_epsilon=config.loss_epsilon)

    for island_idx in range(len(islands)):
        if island_stagnation[island_idx] < config.stagnation_patience:
            continue
        pop_size = len(islands[island_idx])
        if pop_size == 0:
            continue

        n_elite = int(pop_size * config.restart_elite_fraction)
        if config.restart_elite_fraction > 0 and n_elite == 0:
            n_elite = 1
        n_elite = min(max(n_elite, 0), pop_size)

        n_hof = int(pop_size * config.restart_hof_fraction)
        n_hof = min(max(n_hof, 0), max(0, pop_size - n_elite))
        if not hof_entries:
            n_hof = 0

        new_pop, new_fits, new_losses = [], [], []
        new_mses, new_complexities = [], []

        # Preserve elite members
        if n_elite > 0:
            cur_pop = islands[island_idx]
            cur_fits = island_fitnesses[island_idx]
            elite_idx = sorted(range(len(cur_pop)),
                               key=lambda i: cur_fits[i])[:n_elite]
            for idx in elite_idx:
                c = cur_pop[idx].copy()
                if (c.size() > max_size_limit or c.depth() > config.max_depth or
                        not _respects_constraints(c, all_ops,
                                                  config.constraints,
                                                  config.nested_constraints)):
                    c = VariableNode(0)
                c = _maybe_apply_linear_scaling(
                    c,
                    X,
                    y,
                    all_ops,
                    max_size_limit=max_size_limit,
                    max_depth=config.max_depth,
                    constraints=config.constraints,
                    nested_constraints=config.nested_constraints,
                    enabled=config.linear_scaling,
                    loss_type=config.loss_type,
                    huber_delta=config.huber_delta,
                    loss_epsilon=config.loss_epsilon,
                    min_improvement=config.linear_scaling_min_improvement,
                )
                mse, loss, fit, cplx = score_tree(c, X, y, all_ops,
                                                  parsimony_coefficient, **kw)
                new_pop.append(c)
                new_fits.append(fit)
                new_losses.append(loss)
                new_mses.append(mse)
                new_complexities.append(cplx)
                hof.update(c, mse, fit, complexity=cplx)

        # Seed with mutated HoF entries
        for i in range(n_hof):
            entry = hof_entries[i % len(hof_entries)]
            c = entry.node.copy()
            c, _ = mutate_with_info(
                c,
                unary_ops,
                binary_ops,
                n_features,
                max_depth=config.max_depth,
                weights=config.mutation_weights,
                optimize_fn=optimize_constants,
                optimize_kwargs=opt_kwargs,
                rng=rand,
            )
            c = simplify(c)
            if (c.size() > max_size_limit or c.depth() > config.max_depth or
                    not _respects_constraints(c, all_ops, config.constraints,
                                              config.nested_constraints)):
                c = entry.node.copy()
                if (c.size() > max_size_limit or c.depth() > config.max_depth or
                        not _respects_constraints(c, all_ops,
                                                  config.constraints,
                                                  config.nested_constraints)):
                    c = VariableNode(0)
            c = _maybe_optimize_candidate(
                c,
                do_optimize=(rand.random() < config.optimize_prob),
                X=X,
                y=y,
                all_ops=all_ops,
                config=config,
                max_size_limit=max_size_limit,
                constraints=config.constraints,
                nested_constraints=config.nested_constraints,
            )
            mse, loss, fit, cplx = score_tree(c, X, y, all_ops,
                                              parsimony_coefficient, **kw)
            new_pop.append(c)
            new_fits.append(fit)
            new_losses.append(loss)
            new_mses.append(mse)
            new_complexities.append(cplx)
            hof.update(c, mse, fit, complexity=cplx)

        # Fill rest with fresh random trees
        remaining = pop_size - len(new_pop)
        for idx in range(remaining):
            tree = _create_valid_random_tree(
                idx,
                unary_ops,
                binary_ops,
                n_features,
                config.max_depth,
                max_size_limit,
                all_ops,
                config.constraints,
                config.nested_constraints,
                rand,
            )
            tree = _maybe_optimize_candidate(
                tree,
                do_optimize=(rand.random() < config.optimize_prob),
                X=X,
                y=y,
                all_ops=all_ops,
                config=config,
                max_size_limit=max_size_limit,
                constraints=config.constraints,
                nested_constraints=config.nested_constraints,
            )
            mse, loss, fit, cplx = score_tree(tree, X, y, all_ops,
                                              parsimony_coefficient, **kw)
            new_pop.append(tree)
            new_fits.append(fit)
            new_losses.append(loss)
            new_mses.append(mse)
            new_complexities.append(cplx)
            hof.update(tree, mse, fit, complexity=cplx)

        islands[island_idx] = new_pop
        island_fitnesses[island_idx] = new_fits
        island_losses[island_idx] = new_losses
        island_mses[island_idx] = new_mses
        island_complexities[island_idx] = new_complexities
        island_best_losses[island_idx] = min(new_losses)
        island_stagnation[island_idx] = 0


# ---------------------------------------------------------------------------
# Final HoF refinement
# ---------------------------------------------------------------------------


def _final_refine_hof_entries(hof, X, y, all_ops, config,
                              parsimony_coefficient):
    """Run an extra constant-optimization pass on top HoF entries."""
    from deepchem.models.symbolic_regression.evolution.evolve import (
        _maybe_apply_linear_scaling,
        _respects_constraints,
    )

    if config.final_refine_topk <= 0 or not hof.entries:
        return 0

    top_entries = sorted(hof.entries.values(),
                         key=lambda e: e.mse)[:config.final_refine_topk]
    refine_steps = max(config.optimize_steps, 4 * config.optimize_steps)
    refine_restarts = max(config.optimize_n_restarts, 2)
    kw = dict(loss_type=config.loss_type,
              loss_scale=config.loss_scale,
              huber_delta=config.huber_delta,
              loss_epsilon=config.loss_epsilon)

    improvements = 0
    for entry in top_entries:
        candidate = entry.node.copy()
        candidate = optimize_constants(
            candidate,
            X,
            y,
            all_ops,
            steps=refine_steps,
            lr=config.optimize_lr,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            n_restarts=refine_restarts,
            restart_noise=config.optimize_restart_noise,
        )
        candidate = simplify(candidate)
        candidate = _maybe_apply_linear_scaling(
            candidate,
            X,
            y,
            all_ops,
            max_size_limit=config.max_size,
            max_depth=config.max_depth,
            constraints=config.constraints,
            nested_constraints=config.nested_constraints,
            enabled=config.linear_scaling,
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            loss_epsilon=config.loss_epsilon,
            min_improvement=config.linear_scaling_min_improvement,
        )
        if (candidate.size() <= config.max_size and
                candidate.depth() <= config.max_depth and
                _respects_constraints(candidate, all_ops, config.constraints,
                                      config.nested_constraints)):
            mse, loss, fit, cplx = score_tree(candidate, X, y, all_ops,
                                              parsimony_coefficient, **kw)
            prev = hof.entries.get(cplx)
            prev_mse = prev.mse if prev is not None else float("inf")
            if mse + 1e-12 < prev_mse:
                improvements += 1
            hof.update(candidate, mse, fit, complexity=cplx)

    return improvements
