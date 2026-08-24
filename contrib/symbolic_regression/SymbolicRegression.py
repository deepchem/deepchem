import torch
from RegEvolCycle import reg_evol_cycle
from Utils import (
    simplify_tree,
    optimize_constants,
    combine_ops,
)
from LossAndCost import calculate_loss_and_cost


def s_r_cycle(
    dataset,
    population,
    ncycles,
    curmaxsize,
    running_stats,
    options,
):
    """
    use simulated annealing , i follow pysr here that the temperature schedule is linear decay
    """
    if options.annealing and ncycles > 1:
        max_temp = 1.0
        min_temp = float(getattr(options, "temperature_floor", 0.0))
        temps = torch.linspace(max_temp, min_temp, ncycles).tolist()
    else:
        temps = [1.0] * ncycles

    best_by_size = {}

    for T_i, T in enumerate(temps):
        cycle_dataset = dataset
        if getattr(options, "batching", False):
            cycle_dataset = dataset.sample_batch(
                int(getattr(options, "batch_size", 256))
            )

        population = reg_evol_cycle(
            cycle_dataset,
            population,
            temperature=T,
            curmaxsize=curmaxsize,
            running_stats=running_stats,
            options=options,
        )

        for cand in population:
            size = cand.complexity
            if 0 < size <= options.maxsize:
                compare_cand = cand
                # Keep hall-of-fame quality anchored to full-data evaluations.
                if getattr(options, "batching", False):
                    full_every = max(1, int(getattr(options, "full_eval_every", 1)))
                    if (T_i % full_every) == 0:
                        full_loss, full_cost = calculate_loss_and_cost(
                            complexity=cand.complexity,
                            dataset=dataset,
                            tree=cand.tree,
                            parsimony_penalty=options.parsimony_penalty,
                            options=options,
                        )
                        compare_cand = cand.deep_copy()
                        compare_cand.loss = full_loss
                        compare_cand.cost = full_cost
                if (
                    size not in best_by_size
                    or compare_cand.cost < best_by_size[size].cost
                ):
                    best_by_size[size] = compare_cand.deep_copy()

    return population, best_by_size


def optimize_and_simplify_population(dataset, population, options):
    """
    simplify using some rules and optimize constants using gradient descent
    """
    do_opt = torch.rand(len(population)) < options.optimizer_probability

    for i, cand in enumerate(population):
        if options.should_simplify:
            cand.tree = simplify_tree(cand.tree)
            cand.tree = combine_ops(cand.tree)

        if options.should_optimize_constants and do_opt[i]:
            cand = optimize_constants(dataset, cand, options)

        population.candidates[i] = cand

    return population
