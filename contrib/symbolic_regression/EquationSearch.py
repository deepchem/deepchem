from concurrent.futures import ProcessPoolExecutor
import os
from Options import Options
from Population import Population  
from SymbolicRegression import optimize_and_simplify_population, s_r_cycle
from Dataset import Dataset, State
from Utils import get_cur_maxsize, update_hof_from_best_seen, update_hof_from_candidates

def run_s_r_cycle(in_pop:Population,dataset:Dataset, options:Options, cur_maxsize):
    """
        run a single s_r cycle: mutation/crossover, simplification, optimizationr
    """

    out_pop, best_seen = s_r_cycle(
    dataset=dataset,
    population=in_pop,
    ncycles=options.ncycles_per_iteration,
    curmaxsize=cur_maxsize,
    options=options,
    )
    out_pop = optimize_and_simplify_population(dataset, out_pop, options)
    return out_pop, best_seen


def run_population(in_pop:Population,dataset:Dataset, options:Options,total_cycles):
    """
    run mutation or crossover then simplify and optimize the constants
    """
    
    cur_maxsize = options.maxsize
    this_pop_hof = {}
    cur_pop = in_pop

    while total_cycles > 0: 
        out_pop, best_seen = run_s_r_cycle(in_pop,dataset,options,cur_maxsize)
        update_hof_from_candidates(this_pop_hof, out_pop.members, options)
        update_hof_from_best_seen(this_pop_hof, best_seen, options)

        cur_pop = out_pop
        total_cycles -=1 
        cur_maxsize = get_cur_maxsize(options=options,total_cycles=total_cycles,cycles_remaining=total_cycles) # max size is increased as we go


    return cur_pop, this_pop_hof
     

def main_search_loop(state:State, datasets:Dataset, options:Options):
      nout = len(datasets) # get number of targets 

      for j in range(nout):
        dataset = datasets[j]

        total_cycles = options.niterations 
        max_processes_number = min(options.populations, os.cpu_count())

        with ProcessPoolExecutor(max_workers=max_processes_number) as executor: 
            futures = []
            for i in range(options.populations): # run a process for each population
                futures.append(
                    executor.submit(
                        run_population,
                        state.populations[j][i],
                        dataset,
                        options,
                        total_cycles,
                    )
                )

            for i, fut in enumerate(futures):
                out_pop, local_hof = fut.result()
                state.populations = out_pop
                update_hof_from_best_seen(state.hof[j], local_hof, options)

        state.cycles_remain[j] = 0

def equation_search(X, y, niterations, nout):
    """
        nout: support creating equations from dataset to different targets as pysr do.
    """

    options = Options(
        ops=["add", "sub", "mul", "div", "neg", "sin", "cos", "log", "exp"],
        unary_ops=["neg", "sin", "cos", "log", "exp"],
        binary_ops=["add", "sub", "mul", "div"],
        nfeatures=X.shape[1],
        optimizer_probability=0.5,
        tournament_selection_p=0.982,
        temperature_floor=0.05,
        parsimony_penalty=0.01,
        niterations=niterations,  
        populations=8,
        maxsize=12,
        maxdepth=12,
    )

    datasets = [Dataset(X, y) for _ in range(nout)]

    populations = [] # creating pop for each target this is 2d array as for each output run multiple populations in parrallel

    for j in range(nout):
        pop = [
            Population.random_population(
                dataset=datasets[j],
                population_size=options.population_size,
                tree_depth=3,
                options=options,
            )
            for _ in range(options.populations)
        ]
        populations.append(pop)

    total_cycles = niterations * options.populations # run niterations per pop

    state = State(
        populations=populations,                 
        hof=[{} for _ in range(nout)], # create holl-of-frame for each output, hof is a thing pysr use to save each complexity level 
        cycles_remain=[total_cycles for _ in range(nout)],
    )

    main_search_loop(state, datasets, options)
    return state


