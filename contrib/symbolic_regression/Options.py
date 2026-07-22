# ============================================================
# MutationWeightsModule
# ============================================================


class MutationWeights:
    # Match SymbolicRegression.jl defaults for mutation sampling.
    def __init__(
        self,
        mutate_constant=0.0353,
        mutate_operator=3.63,
        mutate_feature=0.1,
        swap_operands=0.00608,
        rotate_tree=1.42,
        add_node=0.0771,
        insert_node=2.44,
        delete_node=0.369,
        simplify=0.00148,
        randomize=0.00695,
        do_nothing=0.431,
        optimize=0.0,
    ):
        self.mutate_constant = mutate_constant
        self.mutate_operator = mutate_operator
        self.mutate_feature = mutate_feature
        self.swap_operands = swap_operands
        self.rotate_tree = rotate_tree
        self.add_node = add_node
        self.insert_node = insert_node
        self.delete_node = delete_node
        self.simplify = simplify
        self.randomize = randomize
        self.do_nothing = do_nothing
        self.optimize = optimize

    def copy(self):
        return MutationWeights(
            mutate_constant=self.mutate_constant,
            mutate_operator=self.mutate_operator,
            mutate_feature=self.mutate_feature,
            swap_operands=self.swap_operands,
            rotate_tree=self.rotate_tree,
            add_node=self.add_node,
            insert_node=self.insert_node,
            delete_node=self.delete_node,
            simplify=self.simplify,
            randomize=self.randomize,
            do_nothing=self.do_nothing,
            optimize=self.optimize,
        )

    def to_vector(self):
        return [
            self.mutate_constant,
            self.mutate_operator,
            self.mutate_feature,
            self.swap_operands,
            self.rotate_tree,
            self.add_node,
            self.insert_node,
            self.delete_node,
            self.simplify,
            self.randomize,
            self.do_nothing,
            self.optimize,
        ]


V_MUTATIONS = [
    "mutate_constant",
    "mutate_operator",
    "mutate_feature",
    "swap_operands",
    "rotate_tree",
    "add_node",
    "insert_node",
    "delete_node",
    "simplify",
    "randomize",
    "do_nothing",
    "optimize",
]


class Options:
    def __init__(self, ops, unary_ops, binary_ops, nfeatures, **kwargs):
        # allowed operators for random generation / mutation later
        self.ops = ops
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.nfeatures = nfeatures

        defaults = {
            # tournament selection
            "tournament_selection_n": 15,
            "tournament_selection_p": 0.982,
            # selection objective
            "parsimony_penalty": 0.0,
            # "log" expects non-negative loss; "linear" allows any real-valued loss.
            "loss_scale": "log",
            # Optional custom loss hook: fn(tree, dataset, options) -> float
            "loss_function": None,
            "use_frequency": True,
            "use_frequency_in_tournament": True,
            "adaptive_parsimony_scaling": 20.0,
            "frequency_window_size": 100000,
            "complexity_of_operators": None,
            "complexity_of_constants": 1.0,
            "complexity_of_variables": 1.0,
            "complexity_mapping": None,
            # evolution controls
            "crossover_probability": 0.0259,
            "max_num_of_mutations": 10,
            "population_size": 27,
            "populations": 5,
            "annealing": True,
            "optimizer_probability": 0.1,
            "should_simplify": True,
            "opt_method": "LBFGS",  # or "Adam"
            "should_optimize_constants": True,
            "opt_lr": 0.1,
            "opt_steps": 200,
            "optimizer_nrestarts": 2,
            "alpha": 1.0,
            "verbose": False,
            "temperature_floor": 0.05,
            "ncycles_per_iteration": 380,
            "warmup_maxsize_by": 0,  # fraction of total cycles
            "niterations": 100,
            "perturbation_factor": 0.129,
            "probability_negate_constant": 0.00743,
            "skip_mutation_failures": True,
            "hof_migration": True,
            "fraction_replaced_hof": 0.0614,
            # IMPORTANT: mutation weights object
            "mutation_weights": MutationWeights(),
            # global constraints
            "maxsize": 30,
            "maxdepth": 100,
            # operator-specific constraints
            "max_child_complexity": None,
            "max_child_depth": None,
            # illegal nesting rules
            "illegal_nesting": None,
            # optional extra constraints
            "forbid_div_by_const": False,
            "forbid_domain_violations": False,
            # How to pick a final equation from hall-of-fame.
            # "pareto_score"/"score": max Pareto score.
            # "min_cost": min normalized cost.
            # "best": score-based selection among near-best-loss equations.
            "model_selection": "pareto_score",
            "topn": 12,
            "best_loss_multiplier": 1.5,
            "parallelism": "async",
            "max_workers": 1,
            "migration": False,
            "fraction_replaced": 0.0,
            "fraction_replaced_guesses": 0.0,
            "seed": None,
            "deterministic": False,
            "timeout_in_seconds": None,
            "max_evals": None,
            # mini-batch evolution
            "batching": False,
            "batch_size": 256,
            "full_eval_every": 1,
            # warm-start / persistence
            "checkpoint_file": None,
            "checkpoint_frequency": 1,
            "return_state": True,
            # richer loss controls
            "loss_name": "mse",  # mse|mae|huber|logcosh|quantile|hinge|bce
            "huber_delta": 1.0,
            "quantile": 0.5,
            "loss_function_expression": None,
            # dimensional analysis
            "enforce_dimensional_constraints": False,
            "dimensional_constraint_penalty": None,
            "dimensionless_constants_only": False,
            # experimental: structured/parametric entry points
            "template_spec": None,
            "class_labels": None,
            "max_parameters": 4,
        }

        allowed = set(defaults.keys())
        for key in kwargs:
            if key not in allowed:
                raise TypeError(
                    "Options.__init__() got an unexpected keyword argument '%s'" % key
                )

        for key, value in defaults.items():
            if key == "mutation_weights":
                setattr(self, key, value.copy())
            else:
                setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
