import random
import numpy as np
from Generation import crossover_generation, next_generation


def reg_evol_cycle(
    dataset,
    pop,
    temperature,
    curmaxsize,
    running_stats,
    options,
):

    n_evol_cycles = np.ceil(
        pop.n / options.tournament_selection_n,
    ).astype(
        int
    )  # as we take the best of just tournament_selection_n candidates i run it many times

    for _ in range(n_evol_cycles):

        if random.random() > options.crossover_probability:

            best_sample = pop.best_of_sample(options, running_stats)

            baby, accepted = next_generation(
                dataset=dataset,
                member=best_sample,
                temperature=temperature,
                curmaxsize=curmaxsize,
                running_stats=running_stats,
                options=options,
            )

            if (not accepted) and options.skip_mutation_failures:
                continue

            oldest = pop.argmin_birth()
            pop.members[oldest] = baby

        else:
            allstar1 = pop.best_of_sample(options, running_stats)
            allstar2 = pop.best_of_sample(options, running_stats)

            result = crossover_generation(
                allstar1,
                allstar2,
                dataset,
                curmaxsize,
                options,
            )

            baby1, baby2, accepted = result

            if (not accepted) and options.skip_mutation_failures:
                print("crossover not accepted")
                continue

            oldest1 = pop.argmin_birth()
            oldest2 = pop.argmin_birth_excluding(exclude_idx=oldest1)

            pop.members[oldest1] = baby1
            pop.members[oldest2] = baby2

    return pop
