from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import os
import pickle
import random
import time
from uuid import uuid4

from Options import Options
from Population import Population
from SymbolicRegression import optimize_and_simplify_population, s_r_cycle
from Dataset import Dataset, State
from AdaptiveParsimony import RunningSearchStatistics
from Utils import (
    get_cur_maxsize,
    update_hof_from_best_seen,
    update_hof_from_candidates,
    calculate_pareto_frontier,
    format_hall_of_fame,
    select_best_candidate,
    migrate_candidates,
)

try:
    import torch
except Exception:
    torch = None


def _default_checkpoint_path(run_id):
    return os.path.join("outputs", run_id, "checkpoint.pkl")


def save_checkpoint(state, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dispatch_s_r_cycle(
    in_pop,
    dataset,
    options,
    cur_maxsize,
    running_stats,
):
    """
    Run one iteration of mutation/crossover then simplification and constant optimization.
    """
    out_pop, best_seen = s_r_cycle(
        dataset=dataset,
        population=in_pop,
        ncycles=options.ncycles_per_iteration,
        curmaxsize=cur_maxsize,
        running_stats=running_stats,
        options=options,
    )
    out_pop = optimize_and_simplify_population(dataset, out_pop, options)
    # We track a rough eval count for stop criteria in this implementation.
    num_evals = float(len(out_pop.members))
    return out_pop, best_seen, num_evals


def _run_population_niterations(
    in_pop,
    dataset,
    options,
    total_cycles,
    running_stats,
):
    """
    Batch mode: run all iterations for one population in one worker task.
    """
    cur_pop = in_pop
    cycles_remaining = total_cycles
    cur_maxsize = options.maxsize
    local_hof = {}
    num_evals = 0.0

    while cycles_remaining > 0:
        out_pop, best_seen, cycle_evals = _dispatch_s_r_cycle(
            in_pop=cur_pop,
            dataset=dataset,
            options=options,
            cur_maxsize=cur_maxsize,
            running_stats=running_stats,
        )
        num_evals += cycle_evals

        update_hof_from_candidates(local_hof, out_pop.members, options)
        update_hof_from_best_seen(local_hof, best_seen, options)

        if running_stats is not None:
            running_stats.update_many([c.complexity for c in out_pop.members])
            running_stats.move_window()
            running_stats.normalize()

        if options.hof_migration:
            dominating = calculate_pareto_frontier(local_hof)
            migrate_candidates(
                dominating, out_pop.members, options.fraction_replaced_hof
            )

        cur_pop = out_pop
        cycles_remaining -= 1
        cur_maxsize = get_cur_maxsize(
            options=options,
            total_cycles=total_cycles,
            cycles_remaining=cycles_remaining,
        )

    return cur_pop, local_hof, running_stats, num_evals


def _run_single_population_iteration(
    pop_idx,
    in_pop,
    dataset,
    options,
    cur_maxsize,
    running_stats,
):
    out_pop, best_seen, num_evals = _dispatch_s_r_cycle(
        in_pop=in_pop,
        dataset=dataset,
        options=options,
        cur_maxsize=cur_maxsize,
        running_stats=running_stats,
    )
    return pop_idx, out_pop, best_seen, num_evals


def _should_stop(state, options, started_at):
    timeout = getattr(options, "timeout_in_seconds", None)
    if timeout is not None and (time.time() - started_at) >= float(timeout):
        return True
    max_evals = getattr(options, "max_evals", None)
    if max_evals is not None and float(state.n_evals) >= float(max_evals):
        return True
    return False


def _apply_migration(state, out_idx, pop_idx, options):
    if (
        getattr(options, "migration", False)
        and getattr(options, "fraction_replaced", 0.0) > 0.0
    ):
        migrants = []
        for i, pop in enumerate(state.last_pops[out_idx]):
            if i == pop_idx or len(pop.members) == 0:
                continue
            migrants.append(min(pop.members, key=lambda c: c.cost))
        migrate_candidates(
            migrants,
            state.last_pops[out_idx][pop_idx].members,
            options.fraction_replaced,
        )

    if options.hof_migration and getattr(options, "fraction_replaced_hof", 0.0) > 0.0:
        dominating = calculate_pareto_frontier(state.halls_of_fame[out_idx])
        migrate_candidates(
            dominating,
            state.last_pops[out_idx][pop_idx].members,
            options.fraction_replaced_hof,
        )


def _main_search_loop_batch(state, datasets, options):
    nout = len(datasets)
    for j in range(nout):
        dataset = datasets[j]
        total_cycles = options.niterations
        max_workers = options.max_workers or min(
            options.populations, os.cpu_count() or 1
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(options.populations):
                futures.append(
                    executor.submit(
                        _run_population_niterations,
                        state.last_pops[j][i],
                        dataset,
                        options,
                        total_cycles,
                        state.running_stats[j].copy() if state.running_stats else None,
                    )
                )

            stats_to_merge = []
            for i, fut in enumerate(futures):
                out_pop, local_hof, out_stats, num_evals = fut.result()
                state.last_pops[j][i] = out_pop
                update_hof_from_best_seen(state.halls_of_fame[j], local_hof, options)
                state.n_evals += num_evals
                state.ncycles_done[j][i] = options.niterations
                if out_stats is not None:
                    stats_to_merge.append(out_stats)

            if stats_to_merge:
                state.running_stats[j] = RunningSearchStatistics.merge(stats_to_merge)

        state.cycles_remaining[j] = 0
        state.cur_maxsizes[j] = options.maxsize


def _main_search_loop_async(state, datasets, options):
    nout = len(datasets)
    started_at = time.time()
    checkpoint_every = max(1, int(getattr(options, "checkpoint_frequency", 1)))
    checkpoint_path = options.checkpoint_file or _default_checkpoint_path(state.run_id)
    completed_since_save = 0

    for j in range(nout):
        dataset = datasets[j]
        total_cycles = options.niterations * options.populations
        max_workers = options.max_workers or min(
            options.populations, os.cpu_count() or 1
        )

        # Low-memory fallback: run one-population step at a time in-process.
        if max_workers <= 1:
            for pop_idx in range(options.populations):
                while state.ncycles_done[j][pop_idx] < options.niterations:
                    if _should_stop(state, options, started_at):
                        break
                    out_pop, best_seen, num_evals = _dispatch_s_r_cycle(
                        in_pop=state.last_pops[j][pop_idx],
                        dataset=dataset,
                        options=options,
                        cur_maxsize=state.cur_maxsizes[j],
                        running_stats=(
                            state.running_stats[j] if state.running_stats else None
                        ),
                    )
                    state.last_pops[j][pop_idx] = out_pop
                    update_hof_from_candidates(
                        state.halls_of_fame[j], out_pop.members, options
                    )
                    update_hof_from_best_seen(
                        state.halls_of_fame[j], best_seen, options
                    )
                    state.n_evals += num_evals
                    if state.running_stats is not None:
                        state.running_stats[j].update_many(
                            [c.complexity for c in out_pop.members]
                        )
                        state.running_stats[j].move_window()
                        state.running_stats[j].normalize()
                    state.ncycles_done[j][pop_idx] += 1
                    state.cycles_remaining[j] = max(0, state.cycles_remaining[j] - 1)
                    state.cur_maxsizes[j] = get_cur_maxsize(
                        options=options,
                        total_cycles=total_cycles,
                        cycles_remaining=state.cycles_remaining[j],
                    )
                    _apply_migration(state, j, pop_idx, options)
                    completed_since_save += 1
                    if (
                        options.checkpoint_file
                        and completed_since_save >= checkpoint_every
                    ):
                        save_checkpoint(state, checkpoint_path)
                        completed_since_save = 0
            state.cycles_remaining[j] = max(
                0,
                options.niterations * options.populations - sum(state.ncycles_done[j]),
            )
            state.cur_maxsizes[j] = (
                options.maxsize
                if state.cycles_remaining[j] == 0
                else state.cur_maxsizes[j]
            )
            continue

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i in range(options.populations):
                if state.ncycles_done[j][i] >= options.niterations:
                    continue
                fut = executor.submit(
                    _run_single_population_iteration,
                    i,
                    state.last_pops[j][i],
                    dataset,
                    options,
                    state.cur_maxsizes[j],
                    state.running_stats[j].copy() if state.running_stats else None,
                )
                futures[fut] = i

            while futures:
                if _should_stop(state, options, started_at):
                    for f in list(futures.keys()):
                        f.cancel()
                    futures.clear()
                    break

                done, _ = wait(
                    list(futures.keys()), timeout=0.2, return_when=FIRST_COMPLETED
                )
                if not done:
                    continue

                for fut in done:
                    failed_pop_idx = futures.pop(fut)
                    try:
                        pop_idx, out_pop, best_seen, num_evals = fut.result()
                    except Exception:
                        # If worker pools fail (common under tight memory), retry in-process.
                        pop_idx = failed_pop_idx
                        out_pop, best_seen, num_evals = _dispatch_s_r_cycle(
                            in_pop=state.last_pops[j][pop_idx],
                            dataset=dataset,
                            options=options,
                            cur_maxsize=state.cur_maxsizes[j],
                            running_stats=(
                                state.running_stats[j] if state.running_stats else None
                            ),
                        )

                    state.last_pops[j][pop_idx] = out_pop
                    update_hof_from_candidates(
                        state.halls_of_fame[j], out_pop.members, options
                    )
                    update_hof_from_best_seen(
                        state.halls_of_fame[j], best_seen, options
                    )
                    state.n_evals += num_evals

                    if state.running_stats is not None:
                        state.running_stats[j].update_many(
                            [c.complexity for c in out_pop.members]
                        )
                        state.running_stats[j].move_window()
                        state.running_stats[j].normalize()

                    state.ncycles_done[j][pop_idx] += 1
                    state.cycles_remaining[j] = max(0, state.cycles_remaining[j] - 1)
                    state.cur_maxsizes[j] = get_cur_maxsize(
                        options=options,
                        total_cycles=total_cycles,
                        cycles_remaining=state.cycles_remaining[j],
                    )

                    _apply_migration(state, j, pop_idx, options)

                    completed_since_save += 1
                    if (
                        options.checkpoint_file
                        and completed_since_save >= checkpoint_every
                    ):
                        save_checkpoint(state, checkpoint_path)
                        completed_since_save = 0

                    if state.ncycles_done[j][pop_idx] >= options.niterations:
                        continue
                    if _should_stop(state, options, started_at):
                        continue

                    next_future = executor.submit(
                        _run_single_population_iteration,
                        pop_idx,
                        state.last_pops[j][pop_idx],
                        dataset,
                        options,
                        state.cur_maxsizes[j],
                        state.running_stats[j].copy() if state.running_stats else None,
                    )
                    futures[next_future] = pop_idx

        state.cycles_remaining[j] = max(
            0,
            options.niterations * options.populations - sum(state.ncycles_done[j]),
        )
        state.cur_maxsizes[j] = (
            options.maxsize if state.cycles_remaining[j] == 0 else state.cur_maxsizes[j]
        )

    if options.checkpoint_file:
        save_checkpoint(state, checkpoint_path)


def main_search_loop(state, datasets, options):
    mode = str(getattr(options, "parallelism", "async")).lower()
    if mode == "batch":
        _main_search_loop_batch(state, datasets, options)
    else:
        _main_search_loop_async(state, datasets, options)


def _clone_options_for_nested_search(options):
    cloned = copy.deepcopy(options)
    cloned.template_spec = None
    cloned.class_labels = None
    return cloned


def template_equation_search(X, y, niterations, *, options):
    """
    Experimental structured search entry-point.
    `options.template_spec` format:
      {
        "slots": {"f": [0, 1], "g": [2]},
        "combine": "add" | "mul" | callable
      }
    """
    spec = options.template_spec or {}
    slots = spec.get("slots", {})
    combine = spec.get("combine", "add")
    if not slots:
        raise ValueError("template_spec.slots is required for template search.")

    slot_states = {}
    slot_preds = {}
    slot_best = {}

    for slot_name, cols in slots.items():
        X_slot = X[:, cols]
        slot_options = _clone_options_for_nested_search(options)
        slot_options.nfeatures = X_slot.shape[1]
        state = equation_search(
            X_slot,
            y,
            niterations,
            nout=1,
            options=slot_options,
        )
        slot_states[slot_name] = state
        best = state.best_candidates[0]
        slot_best[slot_name] = best
        slot_preds[slot_name] = best.tree.forward(X_slot) if best is not None else None

    if callable(combine):
        yhat = combine(slot_preds, X, y)
        combine_str = getattr(combine, "__name__", "custom_combine")
    else:
        if combine == "mul":
            yhat = None
            for v in slot_preds.values():
                if yhat is None:
                    yhat = v
                else:
                    yhat = yhat * v
            combine_str = " * ".join([f"{k}(x)" for k in slot_preds.keys()])
        else:
            yhat = None
            for v in slot_preds.values():
                if yhat is None:
                    yhat = v
                else:
                    yhat = yhat + v
            combine_str = " + ".join([f"{k}(x)" for k in slot_preds.keys()])

    if torch is not None:
        loss = float(((yhat - y) ** 2).mean().detach().cpu().item())
    else:
        yy = list(y)
        yh = list(yhat)
        loss = sum((a - b) ** 2 for a, b in zip(yh, yy)) / max(1, len(yy))

    return {
        "mode": "template",
        "slot_states": slot_states,
        "slot_best": slot_best,
        "combine": combine_str,
        "loss": loss,
    }


def parametric_equation_search(X, y, niterations, *, class_labels, options):
    """
    Experimental class-conditional symbolic search.
    Trains one symbolic model per class and uses class-based dispatch at prediction.
    """
    if class_labels is None:
        raise ValueError("class_labels are required for parametric search.")
    if torch is not None and hasattr(class_labels, "detach"):
        classes = sorted({int(v) for v in class_labels.detach().cpu().tolist()})
    else:
        classes = sorted({int(v) for v in class_labels})

    class_models = {}
    for c in classes:
        if torch is not None and hasattr(class_labels, "detach"):
            idx = class_labels == int(c)
        else:
            idx = [i for i, v in enumerate(class_labels) if int(v) == int(c)]
        Xc = X[idx]
        yc = y[idx]
        cls_options = _clone_options_for_nested_search(options)
        cls_options.nfeatures = Xc.shape[1]
        class_models[int(c)] = equation_search(
            Xc,
            yc,
            niterations,
            nout=1,
            options=cls_options,
        )

    return {
        "mode": "parametric",
        "classes": classes,
        "class_models": class_models,
    }


def _build_initial_state(
    X,
    y,
    niterations,
    nout,
    options,
    *,
    saved_state=None,
):
    if isinstance(saved_state, str):
        state = load_checkpoint(saved_state)
        state.options = options
        if not hasattr(state, "n_evals"):
            state.n_evals = 0.0
        if not hasattr(state, "ncycles_done"):
            state.ncycles_done = [[0 for _ in pops] for pops in state.last_pops]
        if not hasattr(state, "run_id"):
            state.run_id = uuid4().hex[:8]
        return state
    if isinstance(saved_state, State):
        state = saved_state
        state.options = options
        if not hasattr(state, "run_id"):
            state.run_id = uuid4().hex[:8]
        return state

    class_labels = getattr(options, "class_labels", None)
    x_units = getattr(options, "x_units", None)
    y_units = getattr(options, "y_units", None)
    datasets = [
        Dataset(X, y, class_labels=class_labels, x_units=x_units, y_units=y_units)
        for _ in range(nout)
    ]
    running_stats = None
    if options.use_frequency or options.use_frequency_in_tournament:
        running_stats = [
            RunningSearchStatistics(options.maxsize, options.frequency_window_size)
            for _ in range(nout)
        ]

    last_pops = []
    for j in range(nout):
        pops_j = []
        for _ in range(options.populations):
            pops_j.append(
                Population.random_population(
                    dataset=datasets[j],
                    population_size=options.population_size,
                    tree_depth=3,
                    options=options,
                )
            )
        last_pops.append(pops_j)

    return State(
        last_pops=last_pops,
        halls_of_fame=[{} for _ in range(nout)],
        cur_maxsizes=[options.maxsize for _ in range(nout)],
        cycles_remaining=[niterations * options.populations for _ in range(nout)],
        running_stats=running_stats,
        formatted_hofs=[None for _ in range(nout)],
        best_candidates=[None for _ in range(nout)],
        options=options,
        run_id=uuid4().hex[:8],
        n_evals=0.0,
    )


def equation_search(
    X,
    y,
    niterations,
    *,
    nout=1,
    options=None,
    saved_state=None,
    return_state=None,
    **option_overrides,
):
    if options is None:
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
    else:
        options.nfeatures = X.shape[1]
        options.niterations = niterations

    for k, v in option_overrides.items():
        if not hasattr(options, k):
            raise ValueError(f"Unknown option override: {k}")
        setattr(options, k, v)
    if return_state is not None:
        options.return_state = bool(return_state)

    if options.seed is not None:
        random.seed(int(options.seed))
        if torch is not None:
            torch.manual_seed(int(options.seed))
            if bool(getattr(options, "deterministic", False)):
                torch.use_deterministic_algorithms(True)

    if options.tournament_selection_n >= options.population_size:
        raise ValueError("tournament_selection_n must be less than population_size.")

    if options.template_spec is not None:
        return template_equation_search(X, y, niterations, options=options)

    if getattr(options, "class_labels", None) is not None:
        return parametric_equation_search(
            X,
            y,
            niterations,
            class_labels=options.class_labels,
            options=options,
        )

    state = _build_initial_state(
        X, y, niterations, nout, options, saved_state=saved_state
    )
    datasets = [
        Dataset(
            X,
            y,
            class_labels=getattr(options, "class_labels", None),
            x_units=getattr(options, "x_units", None),
            y_units=getattr(options, "y_units", None),
        )
        for _ in range(nout)
    ]

    # If we resumed from checkpoint, preserve remaining counters.
    if saved_state is None:
        state.cycles_remaining = [
            niterations * options.populations for _ in range(nout)
        ]
    else:
        state.cycles_remaining = [
            max(0, niterations * options.populations - sum(state.ncycles_done[j]))
            for j in range(nout)
        ]

    if options.checkpoint_file is None:
        options.checkpoint_file = _default_checkpoint_path(state.run_id)

    main_search_loop(state, datasets, options)

    for j in range(nout):
        state.formatted_hofs[j] = format_hall_of_fame(state.halls_of_fame[j], options)
        state.best_candidates[j] = select_best_candidate(
            state.halls_of_fame[j], options
        )

    if options.checkpoint_file:
        save_checkpoint(state, options.checkpoint_file)

    return state
