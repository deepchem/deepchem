import numpy as np
<<<<<<< HEAD
import deepchem as dc
from deepchem.models.symbolic_regression import SymbolicRegressionModel

# Load a real dataset
tasks, (train, valid, test), transformers = dc.molnet.load_delaney(
    featurizer="rdkit",
    splitter="random",
    transformers=[]
)

def shrink(ds, n_rows=200, n_feats=6, seed=123):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(n_rows, len(ds)), replace=False)
    X = ds.X[idx, :n_feats].astype(np.float32)
    y = ds.y[idx].astype(np.float32).reshape(-1)
    return dc.data.NumpyDataset(X, y)


train_small = shrink(train)
test_small = shrink(test, n_rows=200, n_feats=6, seed=321)




model = SymbolicRegressionModel(
    ops=["add", "mul"],
    unary_ops=[],
    binary_ops=["add", "mul"],
    niterations=6,
    population_size=35,
    populations=1,
    ncycles_per_iteration=30,
    maxsize=8,
    maxdepth=6,
    tree_depth=3,
    optimizer_probability=1.0,
    should_optimize_constants=True,
    opt_steps=80,
    opt_lr=0.1,
    seed=123,
    use_multiprocessing=False,
)

model.fit(train_small)
print("Best equation:", model.get_best_expression())


preds = model.predict(test_small)[:, 0]
mse = np.mean((preds - test_small.y.reshape(-1)) ** 2)
print("Test MSE:", mse)
=======
import pytest
import deepchem as dc
from deepchem.models.symbolic_regression import SymbolicRegressionModel


def _shrink_dataset(dataset, n_rows=200, n_feats=6, seed=123):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=min(n_rows, len(dataset)), replace=False)
    x = dataset.X[idx, :n_feats].astype(np.float32)
    y = dataset.y[idx].astype(np.float32).reshape(-1)
    return dc.data.NumpyDataset(x, y)


def _run_symbolic_regression_example():
    _, (train, _, test), _ = dc.molnet.load_delaney(featurizer="rdkit",
                                                     splitter="random",
                                                     transformers=[])

    train_small = _shrink_dataset(train)
    test_small = _shrink_dataset(test, n_rows=200, n_feats=6, seed=321)

    model = SymbolicRegressionModel(
        ops=["add", "sub", "mul", "div", "neg", "sin", "cos", "log", "exp"],
        unary_ops=["neg", "sin", "cos", "log", "exp"],
        binary_ops=["add", "sub", "mul", "div"],
        niterations=500,
        population_size=27,
        populations=8,
        ncycles_per_iteration=380,
        maxsize=12,
        maxdepth=12,
        parsimony_penalty=0.01,
        tree_depth=3,
        optimizer_probability=0.5,
        should_optimize_constants=True,
        opt_steps=200,
        opt_lr=0.1,
        tournament_selection_p=0.982,
        options_kwargs={"temperature_floor": 0.05},
        seed=123,
        use_multiprocessing=True,
    )

    model.fit(train_small)
    best_expression = model.get_best_expression()
    preds = model.predict(test_small)[:, 0]
    mse = np.mean((preds - test_small.y.reshape(-1))**2)
    return best_expression, preds, test_small, mse


def test_symbolic_regression_model_trains_and_predicts():
    best_expression, preds, test_small, mse = _run_symbolic_regression_example()

    assert isinstance(best_expression, str)
    assert best_expression != ""
    assert preds.shape == (len(test_small),)
    assert np.isfinite(mse)


@pytest.mark.slow
def test_pysr_library_smoke():
    pysr = pytest.importorskip("pysr")

    X = np.linspace(-1.0, 1.0, 30, dtype=np.float64).reshape(-1, 1)
    y = (X[:, 0]**2 + 0.5).astype(np.float64)

    model = pysr.PySRRegressor(
        niterations=1,
        populations=2,
        population_size=8,
        maxsize=10,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        progress=False,
        verbosity=0,
    )

    try:
        model.fit(X, y)
    except Exception as e:
        # PySR requires Julia; in some CI environments the package may be installed
        # while the Julia runtime is unavailable.
        pytest.skip(f"PySR runtime unavailable: {e}")

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


if __name__ == "__main__":
    best_expression, _, _, mse = _run_symbolic_regression_example()
    print("Best equation:", best_expression)
    print("Test MSE:", float(mse))
>>>>>>> add test cases on deepchem data
