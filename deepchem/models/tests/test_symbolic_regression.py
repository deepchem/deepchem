import importlib
import logging
from types import SimpleNamespace

import deepchem as dc
import numpy as np
import pytest
import torch

from deepchem.models.symbolic_regression.core.node import BinaryOpNode, ConstantNode, Node, UnaryOpNode, VariableNode
from deepchem.models.symbolic_regression.core.operators import DEFAULT_BINARY_OPS, DEFAULT_UNARY_OPS
from deepchem.models.symbolic_regression.evolution.evolve import EvolutionConfig, run_evolution
from deepchem.models.symbolic_regression.evolution.hall_of_fame import HallOfFame, HoFEntry
from deepchem.models.symbolic_regression.symbolic_regression_model import SymbolicRegressionModel

srm = importlib.import_module(
    "deepchem.models.symbolic_regression.symbolic_regression_model")


def _base_model_kwargs():
    return dict(
        n_islands=1,
        population_size=2,
        n_cycles=1,
        max_size=5,
        max_depth=3,
        verbose=False,
        final_refine_topk=0,
        optimize_prob=0.0,
    )


def _dataset(n_samples: int = 8):
    X = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32).reshape(-1, 1)
    y = X[:, 0].astype(np.float32)
    return dc.data.NumpyDataset(X, y)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({
            "validation_ratio": 1.0
        }, "validation_ratio"),
        ({
            "selection_method": "bad"
        }, "selection_method"),  # type: ignore[list-item]
        ({
            "validation_ratio": 0.0,
            "selection_method": "accuracy"
        }, "selection_method requires validation_ratio"),
        ({
            "best_loss_factor": 0.0
        }, "best_loss_factor"),
        ({
            "refit_selected_constants": "bad"
        }, "refit_selected_constants"),  # type: ignore[list-item]
        ({
            "warmup_maxsize_by": 0
        }, "warmup_maxsize_by"),
        ({
            "warmup_start_size": 0
        }, "warmup_start_size"),
        ({
            "warmup_start_size": 6,
            "max_size": 5
        }, "warmup_start_size must be <="),
        ({
            "tournament_pick_prob": 0.0
        }, "tournament_pick_prob"),
        ({
            "crossover_prob": -0.1
        }, "crossover_prob"),
        ({
            "offspring_per_cycle": 0
        }, "offspring_per_cycle"),
        ({
            "max_attempts_per_offspring": 0
        }, "max_attempts_per_offspring"),
        ({
            "batching": "bad"
        }, "batching"),  # type: ignore[list-item]
        ({
            "batch_size": 0
        }, "batch_size"),
        ({
            "batching_threshold": 0
        }, "batching_threshold"),
        ({
            "batch_explore_prob": 2.0
        }, "batch_explore_prob"),
        ({
            "optimize_n_restarts": -1
        }, "optimize_n_restarts"),
        ({
            "optimize_restart_noise": -1.0
        }, "optimize_restart_noise"),
        ({
            "optimize_stagnation_boost": -1.0
        }, "optimize_stagnation_boost"),
        ({
            "final_refine_topk": -1
        }, "final_refine_topk"),
        ({
            "restart_elite_fraction": -0.1
        }, "restart_elite_fraction"),
        ({
            "restart_elite_fraction": 1.1
        }, "restart_elite_fraction"),
        ({
            "parsimony_coefficient": -1.0
        }, "parsimony_coefficient"),
        ({
            "loss_type": "bad"
        }, "loss_type"),  # type: ignore[list-item]
        ({
            "loss_scale": "bad"
        }, "loss_scale"),  # type: ignore[list-item]
        ({
            "huber_delta": 0.0
        }, "huber_delta"),
        ({
            "loss_epsilon": 0.0
        }, "loss_epsilon"),
        ({
            "search_seeds": []
        }, "search_seeds must be non-empty"),
        ({
            "search_seeds": [0, "bad"]
        }, "search_seeds must contain only ints"),  # type: ignore[list-item]
        ({
            "unary_operator_names": ["not_an_op"]
        }, "Unknown unary operators"),
        ({
            "unary_operator_names": [],
            "binary_operator_names": []
        }, "At least one unary or binary"),
        ({
            "nested_constraints": []
        }, "nested_constraints"),  # type: ignore[list-item]
    ],
)
@pytest.mark.torch
def test_model_init_guardrails(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        SymbolicRegressionModel(**{**_base_model_kwargs(), **kwargs})


@pytest.mark.torch
def test_model_init_operator_dedup():
    model = SymbolicRegressionModel(
        **_base_model_kwargs(),
        unary_operator_names=["sin", "sin"],
        binary_operator_names=["add", "add"],
    )
    assert list(model._config.unary_ops.keys()) == ["sin"]
    assert list(model._config.binary_ops.keys()) == ["add"]


@pytest.mark.torch
def test_fit_input_validation_branches():
    model = SymbolicRegressionModel(**_base_model_kwargs())

    ds_x_bad = dc.data.NumpyDataset(np.zeros((2, 1, 1), dtype=np.float32),
                                    np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="X must be 1D or 2D"):
        model.fit(ds_x_bad)

    ds_y_bad2 = dc.data.NumpyDataset(
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="single target"):
        model.fit(ds_y_bad2)

    ds_y_bad3 = dc.data.NumpyDataset(
        np.zeros((3, 1), dtype=np.float32),
        np.zeros((3, 1, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="y must be 1D or 2D"):
        model.fit(ds_y_bad3)

    ds_mismatch = dc.data.NumpyDataset(
        np.zeros((3, 1), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="same number of samples"):
        model.fit(ds_mismatch)

    tiny = dc.data.NumpyDataset(np.zeros((2, 1), dtype=np.float32),
                                np.zeros(2, dtype=np.float32))
    model_with_val = SymbolicRegressionModel(**_base_model_kwargs(),
                                             validation_ratio=0.5)
    with pytest.raises(ValueError, match="at least 3 samples"):
        model_with_val.fit(tiny)


@pytest.mark.torch
def test_fit_selection_refit_and_multiseed_logging(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    call_seeds = []

    def fake_run_evolution(X_fit, y_fit, cfg):
        call_seeds.append(cfg.random_seed)
        hof = HallOfFame()
        if cfg.random_seed == 1:
            return hof
        hof.update(VariableNode(0), mse=1.0, fitness=1.0, complexity=1)
        return hof

    def fake_eval_hof(entries, X_val_tensor, y_val_tensor, operators):
        return [
            SimpleNamespace(entry=entries[0],
                            complexity=entries[0].complexity,
                            val_mse=0.1)
        ]

    def fake_select_entry(stats, method="best", best_loss_factor=1.5):
        return stats[0].entry

    def fake_optimize_constants(node, X, y, operators, **kwargs):
        return ConstantNode(0.0)

    def fake_evaluate_tree(node, X, operators):
        if isinstance(node, VariableNode):
            return torch.ones(X.shape[0], dtype=torch.float32)
        return torch.zeros(X.shape[0], dtype=torch.float32)

    monkeypatch.setattr(srm, "run_evolution", fake_run_evolution)
    monkeypatch.setattr(srm, "evaluate_hof_on_validation", fake_eval_hof)
    monkeypatch.setattr(srm, "select_entry", fake_select_entry)
    monkeypatch.setattr(srm, "optimize_constants", fake_optimize_constants)
    monkeypatch.setattr(srm, "evaluate_tree", fake_evaluate_tree)

    model = SymbolicRegressionModel(
        **{
            **_base_model_kwargs(), "verbose": True
        },
        validation_ratio=0.5,
        selection_method="accuracy",
        refit_selected_constants=True,
        search_seeds=[0, 1],
    )
    ds = dc.data.NumpyDataset(
        np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(-1, 1),
        np.zeros(8, dtype=np.float32),
    )
    model.fit(ds)

    assert call_seeds == [0, 1]
    assert isinstance(model._best_entry, HoFEntry)
    assert isinstance(model._best_entry.node, ConstantNode)
    assert "Running 2 seed restarts" in caplog.text
    assert "Seed 0 best MSE" in caplog.text
    assert "Seed 1 produced empty HoF" in caplog.text
    assert "Best expression:" in caplog.text


@pytest.mark.torch
def test_fit_seed_fallback_branches(monkeypatch):
    seen = []

    def fake_run_evolution(X_fit, y_fit, cfg):
        seen.append(cfg.random_seed)
        hof = HallOfFame()
        hof.update(VariableNode(0), mse=0.0, fitness=0.0, complexity=1)
        return hof

    monkeypatch.setattr(srm, "run_evolution", fake_run_evolution)

    model_single = SymbolicRegressionModel(**_base_model_kwargs(),
                                           random_seed=9)
    model_single.fit(_dataset())
    assert seen == [9]

    seen.clear()
    model_default = SymbolicRegressionModel(**_base_model_kwargs())
    model_default.fit(_dataset())
    assert seen == [0]


@pytest.mark.torch
def test_fit_handles_1d_x_and_single_column_y(monkeypatch):
    seen_shapes = []

    def fake_run_evolution(X_fit, y_fit, cfg):
        seen_shapes.append((np.asarray(X_fit).shape, np.asarray(y_fit).shape))
        hof = HallOfFame()
        hof.update(VariableNode(0), mse=0.0, fitness=0.0, complexity=1)
        return hof

    monkeypatch.setattr(srm, "run_evolution", fake_run_evolution)

    model = SymbolicRegressionModel(**_base_model_kwargs(), random_seed=1)
    ds_x1d = dc.data.NumpyDataset(
        np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        np.linspace(-1.0, 1.0, 6, dtype=np.float32),
    )
    model.fit(ds_x1d)
    assert seen_shapes[-1][0] == (6, 1)

    model2 = SymbolicRegressionModel(**_base_model_kwargs(), random_seed=1)
    ds_y2d = dc.data.NumpyDataset(
        np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(-1, 1),
        np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(-1, 1),
    )
    model2.fit(ds_y2d)
    assert seen_shapes[-1][1] == (6,)


@pytest.mark.torch
def test_predict_getters_and_reload_branches(tmp_path):
    model = SymbolicRegressionModel(**_base_model_kwargs(),
                                    model_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="not been fit"):
        model.predict_on_batch(np.zeros((2, 1), dtype=np.float32))

    model._best_entry = HoFEntry(node=VariableNode(0),
                                 mse=0.0,
                                 fitness=0.0,
                                 complexity=1)
    model._n_features = None
    with pytest.raises(RuntimeError, match="missing feature metadata"):
        model.predict_on_batch(np.zeros((2, 1), dtype=np.float32))

    model._n_features = 1
    with pytest.raises(ValueError, match="X must be 1D or 2D"):
        model.predict_on_batch(np.zeros((2, 1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected 1 input features"):
        model.predict_on_batch(np.zeros((2, 2), dtype=np.float32))

    out = model.predict_on_batch(np.array([1.0, 2.0], dtype=np.float32))
    assert out.shape == (2,)

    model_empty = SymbolicRegressionModel(**_base_model_kwargs(),
                                          model_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="not been fit"):
        model_empty.get_equation()
    with pytest.raises(RuntimeError, match="not been fit"):
        model_empty.get_pareto_front()

    model._hof = HallOfFame()
    model._hof.update(VariableNode(0), mse=0.0, fitness=0.0, complexity=1)
    assert isinstance(model.get_equation(), str)
    assert len(model.get_pareto_front()) == 1
    assert model.get_hall_of_fame() is model._hof
    assert model.get_task_type() == "regression"
    assert model.get_num_tasks() == 1

    model.save()

    # Force reload path that infers n_features from best_entry when missing.
    pkl = tmp_path / "symbolic_regression.joblib"
    import joblib
    with pkl.open("rb") as f:
        data = joblib.load(f)
    data["n_features"] = None
    data["best_entry"] = HoFEntry(
        node=BinaryOpNode("add", VariableNode(2), ConstantNode(1.0)),
        mse=0.0,
        fitness=0.0,
        complexity=3,
    )
    joblib.dump(data, pkl)

    reloaded = SymbolicRegressionModel(**_base_model_kwargs(),
                                       model_dir=str(tmp_path))
    reloaded.reload()
    assert reloaded._n_features == 3
    assert SymbolicRegressionModel._infer_n_features(
        UnaryOpNode("sin", VariableNode(2))) == 3
    assert SymbolicRegressionModel._infer_n_features(ConstantNode(1.0)) == 1


# ---------------------------------------------------------------------------
# End-to-end integration tests (merged from test_symbolic_regression_e2e.py)
# ---------------------------------------------------------------------------


def _make_linear_data(n_samples: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    y = (2.0 * X[:, 0] + 1.0).astype(np.float32)
    return X, y


def _has_forbidden_nested(node: Node, outer: str, inner: str) -> bool:
    if isinstance(node, UnaryOpNode):
        if node.op_name == outer and isinstance(node.child, UnaryOpNode):
            if node.child.op_name == inner:
                return True
        return _has_forbidden_nested(node.child, outer, inner)
    if isinstance(node, BinaryOpNode):
        return _has_forbidden_nested(node.left, outer,
                                     inner) or _has_forbidden_nested(
                                         node.right, outer, inner)
    return False


@pytest.mark.torch
def test_run_evolution_smoke_finds_reasonable_solution():
    X, y = _make_linear_data()
    config = EvolutionConfig(
        n_islands=2,
        population_size=24,
        n_cycles=120,
        max_size=10,
        max_depth=5,
        tournament_size=5,
        optimize_prob=0.2,
        optimize_steps=15,
        optimize_lr=0.05,
        batching=False,
        final_refine_topk=4,
        random_seed=0,
        verbose=False,
        avoid_duplicate_offspring=True,
    )
    hof = run_evolution(X, y, config)
    best = hof.get_best()
    assert best is not None
    assert best.mse < 0.2


@pytest.mark.torch
def test_run_evolution_respects_nested_constraints():
    X, y = _make_linear_data(n_samples=64, seed=1)
    unary_ops = {
        "sin": DEFAULT_UNARY_OPS["sin"],
        "neg": DEFAULT_UNARY_OPS["neg"]
    }
    binary_ops = {"add": DEFAULT_BINARY_OPS["add"]}
    config = EvolutionConfig(
        n_islands=2,
        population_size=16,
        n_cycles=40,
        max_size=8,
        max_depth=4,
        optimize_prob=0.0,
        unary_ops=unary_ops,
        binary_ops=binary_ops,
        nested_constraints={"sin": {
            "sin": 0
        }},
        random_seed=2,
        verbose=False,
    )
    hof = run_evolution(X, y, config)
    assert len(hof.entries) > 0
    for entry in hof.entries.values():
        assert not _has_forbidden_nested(entry.node, "sin", "sin")


@pytest.mark.torch
def test_symbolic_model_end_to_end_fit_predict_save_reload(tmp_path):
    X, y = _make_linear_data(n_samples=96, seed=5)
    ds = dc.data.NumpyDataset(X, y)
    model_dir = tmp_path / "sr_model"

    model = SymbolicRegressionModel(
        n_islands=2,
        population_size=24,
        n_cycles=120,
        max_size=10,
        max_depth=5,
        optimize_prob=0.2,
        optimize_steps=15,
        optimize_lr=0.05,
        batching=False,
        final_refine_topk=4,
        search_seeds=[0],
        random_seed=0,
        verbose=False,
        model_dir=str(model_dir),
    )
    model.fit(ds)
    pred_before = model.predict(ds).reshape(-1)
    mse_before = float(np.mean((pred_before - y)**2))
    assert mse_before < 0.2
    assert isinstance(model.get_equation(), str)
    assert len(model.get_pareto_front()) > 0

    model.save()
    reloaded = SymbolicRegressionModel(
        n_islands=1,
        population_size=4,
        n_cycles=1,
        verbose=False,
        model_dir=str(model_dir),
    )
    reloaded.reload()
    pred_after = reloaded.predict(ds).reshape(-1)
    assert np.allclose(pred_before, pred_after, atol=1e-6)
    assert reloaded.get_equation() == model.get_equation()
