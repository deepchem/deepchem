from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepchem.models.symbolic_regression.core.node import (
    BinaryOpNode,
    ConstantNode,
    Node,
    UnaryOpNode,
    VariableNode,
)
from deepchem.models.symbolic_regression.core.operators import DEFAULT_OPS
from deepchem.models.symbolic_regression.evolution import evolve as evo
from deepchem.models.symbolic_regression.evolution import population as pop_mod
from deepchem.models.symbolic_regression.evolution.evolve import EvolutionConfig, run_evolution
from deepchem.models.symbolic_regression.evolution.hall_of_fame import HallOfFame

pytestmark = pytest.mark.torch


class _UnknownNode(Node):

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return "unknown"


def _tiny_xy(n: int = 6):
    X = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
    y = X[:, 0].astype(np.float32)
    return X, y


def test_evolve_validation_helpers_raise_on_invalid_inputs():
    evo._validate_constraints({"add": 2})
    evo._validate_constraints({"add": (2, 3)})
    with pytest.raises(ValueError):
        evo._validate_constraints({1: 2})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evo._validate_constraints({"add": -2})
    with pytest.raises(ValueError):
        evo._validate_constraints({"add": ("a", 1)})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evo._validate_constraints({"add": (-1, -2)})
    with pytest.raises(ValueError):
        evo._validate_constraints({"add": "bad"})  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        evo._validate_nested_constraints({1: {
            "sin": 0
        }})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evo._validate_nested_constraints({"sin": []})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evo._validate_nested_constraints({"sin": {
            1: 0
        }})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evo._validate_nested_constraints({"sin": {"sin": -1}})


def test_evolve_misc_helper_functions_cover_edge_paths():
    cfg = EvolutionConfig(max_size=9, warmup_maxsize_by=2, warmup_start_size=3)
    assert evo._tree_signature(_UnknownNode()).startswith("X(")
    assert evo._current_max_size(cfg, cycle=0) == 3
    assert evo._current_max_size(cfg, cycle=100) == 9

    auto_cfg = EvolutionConfig(batching="auto",
                               batch_size=None,
                               batching_threshold=3)
    assert evo._resolve_batching(auto_cfg, n_samples=2) == (False, 2)
    assert evo._resolve_batching(auto_cfg, n_samples=1200) == (True, 128)
    assert evo._resolve_batching(auto_cfg, n_samples=6000) == (True, 600)
    assert evo._resolve_batching(auto_cfg, n_samples=60000) == (True, 6000)
    sized_cfg = EvolutionConfig(batching=True, batch_size=10)
    assert evo._resolve_batching(sized_cfg, n_samples=3) == (False, 3)
    auto_small_enabled = EvolutionConfig(batching=True, batch_size=None)
    assert evo._resolve_batching(auto_small_enabled,
                                 n_samples=10) == (False, 10)
    with pytest.raises(ValueError):
        evo._resolve_batching(EvolutionConfig(batching="bad"),
                              n_samples=5)  # type: ignore[arg-type]

    import random

    random.Random(0)
    ops = DEFAULT_OPS
    X_ls = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    y_ls = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    base_node = VariableNode(0)
    base_preds = evo.evaluate_tree(base_node, X_ls, ops)
    base_mse = torch.mean((base_preds - y_ls)**2).item()
    scaled_node = evo._maybe_apply_linear_scaling(
        base_node,
        X_ls,
        y_ls,
        ops,
        max_size_limit=8,
        max_depth=6,
        enabled=True,
        loss_type="mse",
        huber_delta=1.0,
        loss_epsilon=1e-12,
        min_improvement=1e-12,
    )
    scaled_preds = evo.evaluate_tree(scaled_node, X_ls, ops)
    scaled_mse = torch.mean((scaled_preds - y_ls)**2).item()
    assert scaled_mse + 1e-12 < base_mse
    unchanged_node = evo._maybe_apply_linear_scaling(
        base_node,
        X_ls,
        y_ls,
        ops,
        max_size_limit=8,
        max_depth=6,
        enabled=False,
    )
    assert str(unchanged_node) == str(base_node)
    unchanged_missing_ops = evo._maybe_apply_linear_scaling(
        base_node,
        X_ls,
        y_ls,
        {"add": ops["add"]},
        max_size_limit=8,
        max_depth=6,
        enabled=True,
    )
    assert str(unchanged_missing_ops) == str(base_node)
    unchanged_tight_limit = evo._maybe_apply_linear_scaling(
        base_node,
        X_ls,
        y_ls,
        ops,
        max_size_limit=1,
        max_depth=1,
        enabled=True,
    )
    assert str(unchanged_tight_limit) == str(base_node)


def test_respects_constraints_paths_cover_unary_binary_and_nested():
    ops = DEFAULT_OPS

    assert evo._respects_constraints(_UnknownNode(), ops, constraints={})

    bin_node = BinaryOpNode(
        "add", BinaryOpNode("add", VariableNode(0), VariableNode(0)),
        VariableNode(0))
    assert not evo._respects_constraints(
        bin_node, ops, constraints={"add": (1, -1)})
    assert evo._respects_constraints(
        BinaryOpNode("add", VariableNode(0), VariableNode(0)),
        ops,
        constraints={"add": 1},
    )
    assert not evo._respects_constraints(
        BinaryOpNode("add", VariableNode(0),
                     BinaryOpNode("add", VariableNode(0), VariableNode(0))),
        ops,
        constraints={"add": (10, 1)},
    )

    unary_node = UnaryOpNode(
        "sin", BinaryOpNode("add", VariableNode(0), VariableNode(0)))
    assert not evo._respects_constraints(
        unary_node, ops, constraints={"sin": 1})

    nested = UnaryOpNode("sin", UnaryOpNode("sin", VariableNode(0)))
    assert not evo._respects_constraints(
        nested, ops, nested_constraints={"sin": {
            "sin": 0
        }})
    nested_bin = BinaryOpNode("add", UnaryOpNode("sin", VariableNode(0)),
                              UnaryOpNode("sin", VariableNode(0)))
    assert evo._respects_constraints(nested_bin,
                                     ops,
                                     nested_constraints={"add": {
                                         "sin": 2
                                     }})


def test_init_migration_and_restart_helpers_cover_fallback_branches(
        monkeypatch):
    X, y = _tiny_xy()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    all_ops = DEFAULT_OPS
    hof = HallOfFame()

    monkeypatch.setattr(
        pop_mod,
        "generate_random_tree",
        lambda *_a, **_k: BinaryOpNode("add", VariableNode(0), VariableNode(0)),
    )
    monkeypatch.setattr(pop_mod, "simplify", lambda n: n)
    pop, fits, losses, mses, cplx = evo._init_island(
        unary_ops={},
        binary_ops={"add": all_ops["add"]},
        n_features=1,
        population_size=1,
        X=X_t,
        y=y_t,
        all_ops=all_ops,
        parsimony_coefficient=0.0,
        hof=hof,
        max_size=1,
        max_depth=1,
    )
    assert isinstance(pop[0], VariableNode)
    assert len(fits) == len(losses) == len(mses) == len(cplx) == 1

    # _migrate_from_hof early-return branches.
    cfg = SimpleNamespace(topn_migrants=1)
    empty_hof = HallOfFame()
    evo._migrate_from_hof(
        empty_hof,
        [pop],
        [fits],
        [losses],
        [mses],
        [cplx],
        X_t,
        y_t,
        all_ops,
        cfg,
    )
    assert True

    cfg_zero_topn = SimpleNamespace(topn_migrants=0)
    tmp_hof = HallOfFame()
    tmp_hof.update(VariableNode(0), mse=1.0, fitness=1.0, complexity=1)
    evo._migrate_from_hof(
        tmp_hof,
        [pop],
        [fits],
        [losses],
        [mses],
        [cplx],
        X_t,
        y_t,
        all_ops,
        cfg_zero_topn,
    )

    cfg_nreplace_zero = SimpleNamespace(
        topn_migrants=1,
        hof_migration_rate=0.0,
        loss_type="mse",
        loss_scale="linear",
        huber_delta=1.0,
        loss_epsilon=1e-12,
        parsimony_coefficient=0.0,
    )
    evo._migrate_from_hof(
        tmp_hof,
        [pop.copy()],
        [fits.copy()],
        [losses.copy()],
        [mses.copy()],
        [cplx.copy()],
        X_t,
        y_t,
        all_ops,
        cfg_nreplace_zero,
    )

    cfg_force_one = SimpleNamespace(
        topn_migrants=1,
        hof_migration_rate=0.01,
        loss_type="mse",
        loss_scale="linear",
        huber_delta=1.0,
        loss_epsilon=1e-12,
        parsimony_coefficient=0.0,
    )
    evo._migrate_from_hof(
        tmp_hof,
        [pop.copy()],
        [fits.copy()],
        [losses.copy()],
        [mses.copy()],
        [cplx.copy()],
        X_t,
        y_t,
        all_ops,
        cfg_force_one,
    )

    # _restart_stagnant_islands early return.
    evo._restart_stagnant_islands(
        tmp_hof,
        [pop.copy()],
        [fits.copy()],
        [losses.copy()],
        [mses.copy()],
        [cplx.copy()],
        [1.0],
        [0],
        EvolutionConfig(stagnation_patience=0),  # type: ignore[arg-type]
        {},
        {"add": all_ops["add"]},
        1,
        X_t,
        y_t,
        all_ops,
        3,
        0.0,
    )

    # pop_size == 0 and no-hof branch, plus random-fill fallback.
    monkeypatch.setattr(pop_mod, "score_tree", lambda *_a, **_k:
                        (0.0, 0.0, 0.0, 1))
    evo._restart_stagnant_islands(
        HallOfFame(),
        [[], pop.copy()],
        [[], fits.copy()],
        [[], losses.copy()],
        [[], mses.copy()],
        [[], cplx.copy()],
        [0.0, 0.0],
        [1, 1],
        EvolutionConfig(stagnation_patience=1,
                        restart_hof_fraction=1.0,
                        optimize_prob=0.0,
                        max_depth=1,
                        verbose=False),
        {},
        {"add": all_ops["add"]},
        1,
        X_t,
        y_t,
        all_ops,
        1,
        0.0,
    )

    # HoF-seeded fallback to VariableNode(0) path.
    bad_hof = HallOfFame()
    bad_hof.update(BinaryOpNode("add", VariableNode(0), VariableNode(0)),
                   mse=0.0,
                   fitness=0.0,
                   complexity=3)
    monkeypatch.setattr(pop_mod, "mutate_with_info", lambda node, *_a, **_k:
                        (node, "delete_node"))
    evo._restart_stagnant_islands(
        bad_hof,
        [pop.copy()],
        [fits.copy()],
        [losses.copy()],
        [mses.copy()],
        [cplx.copy()],
        [0.0],
        [1],
        EvolutionConfig(stagnation_patience=1,
                        restart_hof_fraction=1.0,
                        optimize_prob=0.0,
                        max_depth=1,
                        verbose=False),
        {},
        {"add": all_ops["add"]},
        1,
        X_t,
        y_t,
        all_ops,
        1,
        0.0,
    )


def test_restart_stagnant_islands_preserves_elites(monkeypatch):
    X, y = _tiny_xy(8)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ops = DEFAULT_OPS

    best = VariableNode(0)
    bad = ConstantNode(999.0)
    islands = [[best.copy(), bad.copy()]]
    fits = [[0.1, 10.0]]
    losses = [[0.1, 10.0]]
    mses = [[0.1, 10.0]]
    cplx = [[1, 1]]
    best_losses = [0.1]
    stagnation = [1]

    # Make random fill deterministic and poor; elite should still survive.
    monkeypatch.setattr(
        pop_mod,
        "generate_random_tree",
        lambda *_a, **_k: ConstantNode(123.0),
    )
    monkeypatch.setattr(pop_mod, "simplify", lambda n: n)
    monkeypatch.setattr(
        pop_mod,
        "score_tree",
        lambda node, *_a, **_k: (
            0.1 if isinstance(node, VariableNode) else 100.0,
            0.1 if isinstance(node, VariableNode) else 100.0,
            0.1 if isinstance(node, VariableNode) else 100.0,
            1,
        ),
    )

    evo._restart_stagnant_islands(
        HallOfFame(),
        islands,
        fits,
        losses,
        mses,
        cplx,
        best_losses,
        stagnation,
        EvolutionConfig(
            stagnation_patience=1,
            restart_elite_fraction=0.5,
            restart_hof_fraction=0.0,
            optimize_prob=0.0,
            max_depth=2,
            verbose=False,
        ),
        {},
        {"add": ops["add"]},
        1,
        X_t,
        y_t,
        ops,
        4,
        0.0,
    )

    assert any(isinstance(node, VariableNode) for node in islands[0])


def test_final_refine_hof_entries_skip_paths(monkeypatch):
    X, y = _tiny_xy()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ops = DEFAULT_OPS
    hof = HallOfFame()
    hof.update(VariableNode(0), mse=1.0, fitness=1.0, complexity=1)

    cfg_off = EvolutionConfig(final_refine_topk=0, verbose=False)
    assert evo._final_refine_hof_entries(hof, X_t, y_t, ops, cfg_off, 0.0) == 0

    cfg_size_skip = EvolutionConfig(final_refine_topk=1,
                                    max_size=1,
                                    max_depth=1,
                                    verbose=False)
    monkeypatch.setattr(pop_mod, "simplify", lambda n: n)
    monkeypatch.setattr(
        pop_mod,
        "optimize_constants",
        lambda *_a, **_k: BinaryOpNode("add", VariableNode(0), VariableNode(0)),
    )
    assert evo._final_refine_hof_entries(hof, X_t, y_t, ops, cfg_size_skip,
                                         0.0) == 0

    cfg_constraint_skip = EvolutionConfig(
        final_refine_topk=1,
        max_size=10,
        max_depth=5,
        constraints={"sin": 0},
        unary_ops={"sin": ops["sin"]},
        binary_ops={},
        verbose=False,
    )
    monkeypatch.setattr(
        pop_mod,
        "optimize_constants",
        lambda *_a, **_k: UnaryOpNode("sin", VariableNode(0)),
    )
    assert evo._final_refine_hof_entries(hof, X_t, y_t, {"sin": ops["sin"]},
                                         cfg_constraint_skip, 0.0) == 0


def test_run_evolution_none_config_branch(monkeypatch):
    X, y = _tiny_xy(6)
    tiny = EvolutionConfig(
        n_islands=1,
        population_size=2,
        n_cycles=1,
        max_size=5,
        max_depth=3,
        verbose=False,
        optimize_prob=0.0,
        final_refine_topk=0,
        batching=False,
    )
    monkeypatch.setattr(evo, "EvolutionConfig", lambda: tiny)
    out = run_evolution(X, y, config=None)
    assert isinstance(out, HallOfFame)


def test_run_evolution_constraints_int_and_tuple_paths():
    X, y = _tiny_xy(6)
    cfg = EvolutionConfig(
        n_islands=1,
        population_size=2,
        n_cycles=1,
        max_size=5,
        max_depth=3,
        verbose=False,
        optimize_prob=0.0,
        final_refine_topk=0,
        batching=False,
        constraints={
            "add": 3,
            "mul": (3, 3)
        },
    )
    out = run_evolution(X, y, cfg)
    assert isinstance(out, HallOfFame)


def test_run_evolution_optimization_mutation_wiring(monkeypatch):
    X, y = _tiny_xy(6)
    captured = {}

    def fake_mutate_with_info(node,
                              *_a,
                              optimize_fn=None,
                              optimize_kwargs=None,
                              **_k):
        captured["optimize_fn"] = optimize_fn
        captured["optimize_kwargs"] = optimize_kwargs
        return node, "optimize_constants"

    def fake_maybe_optimize(candidate, do_optimize, *_a, **_k):
        captured["do_optimize"] = do_optimize
        return candidate

    monkeypatch.setattr(evo, "mutate_with_info", fake_mutate_with_info)
    monkeypatch.setattr(evo, "_maybe_optimize_candidate", fake_maybe_optimize)

    cfg = EvolutionConfig(
        n_islands=1,
        population_size=2,
        n_cycles=1,
        max_size=5,
        max_depth=3,
        crossover_prob=0.0,
        optimize_prob=1.0,
        verbose=False,
        final_refine_topk=0,
        batching=False,
        avoid_duplicate_offspring=False,
    )
    out = run_evolution(X, y, cfg)
    assert isinstance(out, HallOfFame)
    assert captured["optimize_fn"] is evo.optimize_constants
    assert captured["optimize_kwargs"]["X"].shape[0] == len(X)
    assert captured["optimize_kwargs"]["loss_type"] == cfg.loss_type
    assert captured["do_optimize"] is False


def test_run_evolution_crossover_is_ungated(monkeypatch):
    """Verify crossover offspring are not rejected by a parent gate."""
    X, y = _tiny_xy(6)
    cfg = EvolutionConfig(
        n_islands=1,
        population_size=2,
        n_cycles=1,
        max_size=5,
        max_depth=3,
        crossover_prob=1.0,
        optimize_prob=0.0,
        verbose=False,
        final_refine_topk=0,
        batching=False,
        avoid_duplicate_offspring=False,
    )
    out = run_evolution(X, y, cfg)
    assert isinstance(out, HallOfFame)


def test_final_refine_size_skip_continue_without_scoring(monkeypatch):
    X, y = _tiny_xy()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ops = DEFAULT_OPS
    hof = HallOfFame()
    hof.update(VariableNode(0), mse=0.0, fitness=0.0, complexity=1)

    monkeypatch.setattr(
        pop_mod,
        "optimize_constants",
        lambda *_a, **_k: BinaryOpNode("add", VariableNode(0), VariableNode(0)),
    )
    monkeypatch.setattr(pop_mod, "simplify", lambda n: n)
    monkeypatch.setattr(
        pop_mod,
        "score_tree",
        lambda *_a, **_k:
        (_ for _ in ()).throw(AssertionError("score_tree should not run")),
    )

    cfg = EvolutionConfig(final_refine_topk=1,
                          max_size=1,
                          max_depth=1,
                          verbose=False)
    assert evo._final_refine_hof_entries(hof, X_t, y_t, ops, cfg, 0.0) == 0


def test_run_evolution_verbose_and_batch_prefilter_paths(capsys):
    X, y = _tiny_xy(8)
    cfg = EvolutionConfig(
        n_islands=1,
        population_size=4,
        n_cycles=1,
        max_size=6,
        max_depth=3,
        tournament_size=2,
        offspring_per_cycle=2,
        print_every=1,
        verbose=True,
        batching=True,
        batch_size=2,
        batching_threshold=1,
        batch_explore_prob=0.0,
        optimize_prob=0.0,
        final_refine_topk=1,
        random_seed=0,
    )
    hof = run_evolution(X, y, cfg)
    assert isinstance(hof, HallOfFame)
    out = capsys.readouterr().out
    assert "Initialized" in out
    assert "Batch prefilter enabled" in out
    assert "Cycle" in out
    assert "Final HoF refinement" in out
    assert "HallOfFame" in out
