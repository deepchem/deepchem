<<<<<<< HEAD
=======
"""
DeepChem wrapper for the local symbolic regression implementation.
"""

>>>>>>> add test cases on deepchem data
import copy
import random
import sys
from pathlib import Path
<<<<<<< HEAD
import numpy as np
import torch
from deepchem.data import Dataset
from deepchem.models.models import Model
from deepchem.utils.data_utils import load_from_disk, save_to_disk



def default_symbolic_regression_path():
    here = Path(__file__).resolve()
    candidate = here.parents[2] / "contrib" / "symbolic_regression"
    if candidate.exists():
        return candidate
    return None




=======
from uuid import uuid4

import numpy as np
import torch
from deepchem.models.models import Model
from deepchem.utils.data_utils import load_from_disk, save_to_disk

def default_symbolic_regression_path():
    here = Path(__file__).resolve()
    candidates = [
        # Current DeepChem repo layout.
        here.parents[2] / "contrib" / "symbolic_regression",
        # Legacy standalone checkout layout.
        here.parents[3] / "symbolic_regression",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


>>>>>>> add test cases on deepchem data
class SymbolicRegressionModel(Model):

    def __init__(
        self,
        *,
        ops=None,
        unary_ops=None,
        binary_ops=None,
        niterations=200,
        population_size=30,
        populations=1,
        ncycles_per_iteration=40,
        maxsize=12,
        maxdepth=12,
        tree_depth=3,
        parsimony_penalty=0.0,
        optimizer_probability=0.5,
        should_optimize_constants=True,
        opt_method="LBFGS",
        opt_steps=100,
        opt_lr=0.1,
<<<<<<< HEAD
=======
        opt_nrestarts=2,
>>>>>>> add test cases on deepchem data
        tournament_selection_n=15,
        tournament_selection_p=0.982,
        annealing=True,
        seed_simple_trees=True,
        seed=None,
        device=None,
        dtype=None,
        options=None,
        options_kwargs=None,
        symbolic_regression_path=None,
        use_multiprocessing=False,
        model_dir=None,
        **kwargs,
    ):
        super(SymbolicRegressionModel, self).__init__(model=self, model_dir=model_dir, **kwargs)

        self.ops = list(ops) if ops is not None else None
        self.unary_ops = list(unary_ops) if unary_ops is not None else None
        self.binary_ops = list(binary_ops) if binary_ops is not None else None
        self.niterations = int(niterations)
        self.population_size = int(population_size)
        self.populations = int(populations)
        self.ncycles_per_iteration = int(ncycles_per_iteration)
        self.maxsize = int(maxsize)
        self.maxdepth = int(maxdepth)
        self.tree_depth = int(tree_depth)
        self.parsimony_penalty = float(parsimony_penalty)
        self.optimizer_probability = float(optimizer_probability)
        self.should_optimize_constants = bool(should_optimize_constants)
        self.opt_method = str(opt_method)
        self.opt_steps = int(opt_steps)
        self.opt_lr = float(opt_lr)
<<<<<<< HEAD
=======
        self.opt_nrestarts = int(opt_nrestarts)
>>>>>>> add test cases on deepchem data
        self.tournament_selection_n = int(tournament_selection_n)
        self.tournament_selection_p = float(tournament_selection_p)
        self.annealing = bool(annealing)
        self.seed_simple_trees = bool(seed_simple_trees)
        self.seed = seed
        self.device = device or "cpu"
        self.dtype = dtype
        self.options = options
        self.options_kwargs = options_kwargs or {}
        self.symbolic_regression_path = symbolic_regression_path
        self.use_multiprocessing = bool(use_multiprocessing)
<<<<<<< HEAD
=======

>>>>>>> add test cases on deepchem data
        self._sr_imports = None
        self._options = None
        self._hall_of_fame = None
        self._best_candidate = None



    def ensure_symbolic_regression(self):
        if self._sr_imports is not None:
            return self._sr_imports

<<<<<<< HEAD
        sr_path = None
        if self.symbolic_regression_path is not None:
            sr_path = Path(self.symbolic_regression_path)
        else:
            sr_path = default_symbolic_regression_path()

=======
        if self.symbolic_regression_path is not None:
            sr_path = Path(self.symbolic_regression_path).expanduser().resolve()
        else:
            sr_path = default_symbolic_regression_path()

        if sr_path is None or not sr_path.is_dir():
            raise ModuleNotFoundError(
                "Unable to locate symbolic regression backend. "
                "Set `symbolic_regression_path` to a directory containing "
                "`Candidate.py`, `Population.py`, and related modules "
                "(e.g. `<repo>/contrib/symbolic_regression`)."
            )

>>>>>>> add test cases on deepchem data
        sr_path_str = str(sr_path)
        if sr_path_str not in sys.path:
            sys.path.insert(0, sr_path_str)

<<<<<<< HEAD
        from Candidate import Candidate, Tree 
        from Dataset import Dataset as SRDataset 
        from Options import Options 
        from Population import Population 
        from SymbolicRegression import ( 
            optimize_and_simplify_population,
            s_r_cycle,
        )
        from Utils import ( 
            get_cur_maxsize,
            update_hof_from_best_seen,
            update_hof_from_candidates,
        )
        from equation_search import ( 
            State,
            main_search_loop,
        )
=======
        try:
            from Candidate import Candidate
            from TreeDS import Tree
            from Dataset import Dataset as SRDataset
            from Options import Options
            from Population import Population
            from SymbolicRegression import (
                optimize_and_simplify_population,
                s_r_cycle,
            )
            from AdaptiveParsimony import RunningSearchStatistics
            from Utils import (
                get_cur_maxsize,
                update_hof_from_best_seen,
                update_hof_from_candidates,
                select_best_candidate,
            )
            from equation_search import (
                State,
                main_search_loop,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Failed to import symbolic regression backend from "
                f"`{sr_path}`. Ensure that directory includes the expected "
                "modules such as `Candidate.py` and `SymbolicRegression.py`."
            ) from e
>>>>>>> add test cases on deepchem data

        self._sr_imports = {
            "Candidate": Candidate,
            "Tree": Tree,
            "SRDataset": SRDataset,
            "Options": Options,
            "Population": Population,
            "optimize_and_simplify_population": optimize_and_simplify_population,
            "s_r_cycle": s_r_cycle,
<<<<<<< HEAD
            "get_cur_maxsize": get_cur_maxsize,
            "update_hof_from_best_seen": update_hof_from_best_seen,
            "update_hof_from_candidates": update_hof_from_candidates,
=======
            "RunningSearchStatistics": RunningSearchStatistics,
            "get_cur_maxsize": get_cur_maxsize,
            "update_hof_from_best_seen": update_hof_from_best_seen,
            "update_hof_from_candidates": update_hof_from_candidates,
            "select_best_candidate": select_best_candidate,
>>>>>>> add test cases on deepchem data
            "State": State,
            "main_search_loop": main_search_loop,
        }
        return self._sr_imports

    def _set_seed(self):
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _prepare_dataset(self, dataset):
        X = np.asarray(dataset.X)
        y = np.asarray(dataset.y)
        w = getattr(dataset, "w", None)
        if w is not None:
            w = np.asarray(w)

        if w is not None:
            if w.ndim == 2 and w.shape[1] == 1:
                w = w[:, 0]
<<<<<<< HEAD
=======
            if w.ndim != 1:
                raise ValueError("Sample weights must be 1D or shape (n_samples, 1).")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        if w is not None and len(w) != len(y):
            raise ValueError("Weights must have the same number of samples as y.")
>>>>>>> add test cases on deepchem data

        return X, y, w

    def _infer_ops(self, ops, Tree):
        unary_ops = [op for op in ops if Tree._arity(op) == 1]
        binary_ops = [op for op in ops if Tree._arity(op) == 2]
        return unary_ops, binary_ops

    def _build_options(self, nfeatures):
        sr = self.ensure_symbolic_regression()
        Options = sr["Options"]
        Tree = sr["Tree"]

        default_ops = [
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "sin",
            "cos",
            "log",
            "exp",
        ]

        if self.options is not None:
            options = copy.deepcopy(self.options)
        else:
            ops = self.ops or default_ops
            unary_ops, binary_ops = self._infer_ops(ops, Tree)
            if self.unary_ops is not None:
                unary_ops = list(self.unary_ops)
            if self.binary_ops is not None:
                binary_ops = list(self.binary_ops)

            options = Options(
                ops=ops,
                unary_ops=unary_ops,
                binary_ops=binary_ops,
                nfeatures=nfeatures,
                niterations=self.niterations,
                population_size=self.population_size,
                populations=self.populations,
                ncycles_per_iteration=self.ncycles_per_iteration,
                maxsize=self.maxsize,
                maxdepth=self.maxdepth,
                parsimony_penalty=self.parsimony_penalty,
                optimizer_probability=self.optimizer_probability,
                should_optimize_constants=self.should_optimize_constants,
                opt_method=self.opt_method,
                opt_steps=self.opt_steps,
                opt_lr=self.opt_lr,
<<<<<<< HEAD
=======
                optimizer_nrestarts=self.opt_nrestarts,
>>>>>>> add test cases on deepchem data
                tournament_selection_n=self.tournament_selection_n,
                tournament_selection_p=self.tournament_selection_p,
                annealing=self.annealing,
            )

        options.nfeatures = nfeatures

        if self.options is None:
            overrides = {
                "ops": self.ops,
                "unary_ops": self.unary_ops,
                "binary_ops": self.binary_ops,
                "niterations": self.niterations,
                "population_size": self.population_size,
                "populations": self.populations,
                "ncycles_per_iteration": self.ncycles_per_iteration,
                "maxsize": self.maxsize,
                "maxdepth": self.maxdepth,
                "parsimony_penalty": self.parsimony_penalty,
                "optimizer_probability": self.optimizer_probability,
                "should_optimize_constants": self.should_optimize_constants,
                "opt_method": self.opt_method,
                "opt_steps": self.opt_steps,
                "opt_lr": self.opt_lr,
<<<<<<< HEAD
=======
                "optimizer_nrestarts": self.opt_nrestarts,
>>>>>>> add test cases on deepchem data
                "tournament_selection_n": self.tournament_selection_n,
                "tournament_selection_p": self.tournament_selection_p,
                "annealing": self.annealing,
            }

            for key, value in overrides.items():
                if value is None:
                    continue
                if hasattr(options, key):
                    setattr(options, key, value)

        for key, value in self.options_kwargs.items():
<<<<<<< HEAD
=======
            if not hasattr(options, key):
                raise ValueError(f"Unknown options field: {key}")
>>>>>>> add test cases on deepchem data
            setattr(options, key, value)

        if options.tournament_selection_n > options.population_size:
            options.tournament_selection_n = options.population_size

        return options

    def _build_initial_population(self, dataset, options):
        sr = self.ensure_symbolic_regression()
        Population = sr["Population"]
        return Population.random_population(
            dataset=dataset,
            population_size=options.population_size,
            tree_depth=self.tree_depth,
            options=options,
        )

<<<<<<< HEAD
    def _run_population_niterations(self,dataset,options):
=======
    def _run_population_niterations(self, dataset, options):
>>>>>>> add test cases on deepchem data
        sr = self.ensure_symbolic_regression()
        s_r_cycle = sr["s_r_cycle"]
        optimize_and_simplify_population = sr["optimize_and_simplify_population"]
        update_hof_from_candidates = sr["update_hof_from_candidates"]
        update_hof_from_best_seen = sr["update_hof_from_best_seen"]
        get_cur_maxsize = sr["get_cur_maxsize"]
<<<<<<< HEAD

        cur_pop = self._build_initial_population(dataset, options)
        hof = {}
=======
        RunningSearchStatistics = sr["RunningSearchStatistics"]

        cur_pop = self._build_initial_population(dataset, options)
        hof = {}
        running_stats = None
        if options.use_frequency or options.use_frequency_in_tournament:
            running_stats = RunningSearchStatistics(
                options.maxsize,
                options.frequency_window_size,
            )
>>>>>>> add test cases on deepchem data

        if options.should_optimize_constants:
            cur_pop = optimize_and_simplify_population(dataset, cur_pop, options)
        update_hof_from_candidates(hof, cur_pop.members, options)

        cycles_remaining = options.niterations
        cur_maxsize = options.maxsize

        while cycles_remaining > 0:
            cur_pop, best_seen = s_r_cycle(
                dataset=dataset,
                population=cur_pop,
                ncycles=options.ncycles_per_iteration,
                curmaxsize=cur_maxsize,
<<<<<<< HEAD
=======
                running_stats=running_stats,
>>>>>>> add test cases on deepchem data
                options=options,
            )
            cur_pop = optimize_and_simplify_population(dataset, cur_pop, options)
            update_hof_from_candidates(hof, cur_pop.members, options)
            update_hof_from_best_seen(hof, best_seen, options)
<<<<<<< HEAD
=======
            if running_stats is not None:
                running_stats.update_many([c.complexity for c in cur_pop.members])
                running_stats.move_window()
                running_stats.normalize()
>>>>>>> add test cases on deepchem data

            cycles_remaining -= 1
            cur_maxsize = get_cur_maxsize(
                options=options,
                total_cycles=options.niterations,
                cycles_remaining=cycles_remaining,
            )

<<<<<<< HEAD


        return cur_pop, hof
    

    

    def _run_sequential(self,dataset,options):
=======
        return cur_pop, hof

    def _run_sequential(self, dataset, options):
>>>>>>> add test cases on deepchem data
        sr = self.ensure_symbolic_regression()
        update_hof_from_best_seen = sr["update_hof_from_best_seen"]

        hall_of_fame = {}
        for _ in range(options.populations):
            _, local_hof = self._run_population_niterations(dataset, options)
            update_hof_from_best_seen(hall_of_fame, local_hof, options)
        return hall_of_fame

<<<<<<< HEAD


    def _run_multiprocessing(self,dataset,options):
=======
    def _run_multiprocessing(self, dataset, options):
>>>>>>> add test cases on deepchem data
        sr = self.ensure_symbolic_regression()
        Population = sr["Population"]
        State = sr["State"]
        main_search_loop = sr["main_search_loop"]
        update_hof_from_best_seen = sr["update_hof_from_best_seen"]

        last_pops = []
        pops_j = []
        for _ in range(options.populations):
            pops_j.append(
                Population.random_population(
                    dataset=dataset,
                    population_size=options.population_size,
                    tree_depth=self.tree_depth,
                    options=options,
                )
            )
        last_pops.append(pops_j)

        state = State(
            last_pops=last_pops,
            halls_of_fame=[{}],
            cur_maxsizes=[options.maxsize],
            cycles_remaining=[options.niterations * options.populations],
<<<<<<< HEAD
=======
            run_id=uuid4().hex[:8],
>>>>>>> add test cases on deepchem data
        )

        main_search_loop(state, [dataset], options)
        hall_of_fame = {}
        update_hof_from_best_seen(hall_of_fame, state.halls_of_fame[0], options)
        return hall_of_fame

<<<<<<< HEAD
    def _select_best_candidate(self,hall_of_fame):
        return min(hall_of_fame.values(), key=lambda c: c.cost)
    

=======
    def _select_best_candidate(self, hall_of_fame):
        if not hall_of_fame:
            raise ValueError("No candidates found in hall of fame.")
        sr = self.ensure_symbolic_regression()
        selector = sr.get("select_best_candidate")
        if selector is not None and self._options is not None:
            selected = selector(hall_of_fame, self._options)
            if selected is not None:
                return selected
        return min(hall_of_fame.values(), key=lambda c: c.cost)
>>>>>>> add test cases on deepchem data

    def fit(self, dataset):
        self._set_seed()

        X, y, w = self._prepare_dataset(dataset)
        options = self._build_options(nfeatures=X.shape[1])
        self._options = options

        device = torch.device(self.device)
<<<<<<< HEAD
        dtype = self.dtype or torch.float32
=======
        # PySR/Julia commonly runs with Float64; keep that as our default.
        dtype = self.dtype or torch.float64
>>>>>>> add test cases on deepchem data

        X_t = torch.tensor(X, device=device, dtype=dtype)
        y_t = torch.tensor(y, device=device, dtype=dtype)
        w_t = None
        if w is not None:
            w_t = torch.tensor(w, device=device, dtype=dtype)

        sr = self.ensure_symbolic_regression()
        SRDataset = sr["SRDataset"]
        sr_dataset = SRDataset(X_t, y_t, weights=w_t)

<<<<<<< HEAD
        if self.use_multiprocessing and options.populations > 1:
            hall_of_fame = self._run_multiprocessing(sr_dataset, options)
        else:
            hall_of_fame = self._run_sequential(sr_dataset, options)
=======
        # Prefer the stateful search loop so behavior matches the backend's
        # async/batch orchestration (including hall-of-fame migration logic).
        hall_of_fame = self._run_multiprocessing(sr_dataset, options)
>>>>>>> add test cases on deepchem data

        self._hall_of_fame = hall_of_fame
        self._best_candidate = self._select_best_candidate(hall_of_fame)

<<<<<<< HEAD



    def predict_on_batch(self, X):

        X_arr = np.asarray(X)

        device = torch.device(self.device)
        dtype = self.dtype or torch.float32
=======
    def predict_on_batch(self, X):
        if self._best_candidate is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("SymbolicRegressionModel requires 2D feature arrays.")

        device = torch.device(self.device)
        dtype = self.dtype or torch.float64
>>>>>>> add test cases on deepchem data
        X_t = torch.tensor(X_arr, device=device, dtype=dtype)

        with torch.no_grad():
            y_t = self._best_candidate.tree.forward(X_t)

        y_np = y_t.detach().cpu().numpy().reshape(-1, 1)
        return y_np

    def save(self):
        state = {
            "options": self._options,
            "hall_of_fame": self._hall_of_fame,
            "best_candidate": self._best_candidate,
            "config": {
                "device": self.device,
                "dtype": self.dtype,
                "seed": self.seed,
                "tree_depth": self.tree_depth,
                "seed_simple_trees": self.seed_simple_trees,
            },
        }
        save_to_disk(state, self.get_model_filename(self.model_dir))

<<<<<<< HEAD


=======
>>>>>>> add test cases on deepchem data
    def reload(self):
        state = load_from_disk(self.get_model_filename(self.model_dir))
        self.ensure_symbolic_regression()
        self._options = state.get("options", None)
        self._hall_of_fame = state.get("hall_of_fame", None)
        self._best_candidate = state.get("best_candidate", None)
        config = state.get("config", {})
        if config:
            self.device = config.get("device", self.device)
            self.dtype = config.get("dtype", self.dtype)
            self.seed = config.get("seed", self.seed)
            self.tree_depth = config.get("tree_depth", self.tree_depth)
            self.seed_simple_trees = config.get("seed_simple_trees", self.seed_simple_trees)

<<<<<<< HEAD


=======
>>>>>>> add test cases on deepchem data
    def get_task_type(self):
        return "regression"

    def get_num_tasks(self):
        return 1

    def get_best_expression(self):
        if self._best_candidate is None:
            return None
        return self._best_candidate.tree.to_string()
