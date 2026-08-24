#!/usr/bin/env python3
"""
Compare symbolic regression models on MoleculeNet regression datasets.

This script is intentionally minimal and uses DeepChem's MoleculeNet loaders so both
models see the same train/valid/test splits. It also supports feature slicing to keep
symbolic regression feasible on high-dimensional featurizations.
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import deepchem as dc
from deepchem.models.symbolic_regression import SymbolicRegressionModel
from sklearn.impute import SimpleImputer


Loader = Callable[..., Tuple[List[str], Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset], List[dc.trans.Transformer]]]


REGRESSION_LOADERS: Dict[str, Loader] = {
    "delaney": dc.molnet.load_delaney,
    "freesolv": dc.molnet.load_sampl,
    "lipo": dc.molnet.load_lipo,
    "bace_r": dc.molnet.load_bace_regression,
    "thermosol": dc.molnet.load_thermosol,
}


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def split_ops_for_pysr(ops: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    if not ops:
        return ["+", "-", "*", "/"], ["neg", "sin", "cos", "log", "exp"]

    binary_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    unary_map = {"neg": "neg", "sin": "sin", "cos": "cos", "log": "log", "exp": "exp"}

    binary_ops: List[str] = []
    unary_ops: List[str] = []
    for op in ops:
        if op in binary_map:
            binary_ops.append(binary_map[op])
        elif op in unary_map:
            unary_ops.append(unary_map[op])
        else:
            # Allow passing native PySR operators directly.
            if op in {"+", "-", "*", "/"}:
                binary_ops.append(op)
            else:
                unary_ops.append(op)

    if not binary_ops:
        binary_ops = ["+", "-", "*", "/"]
    return binary_ops, unary_ops


def select_single_task(dataset: dc.data.Dataset, task_index: int = 0) -> dc.data.NumpyDataset:
    X = np.asarray(dataset.X)
    y = np.asarray(dataset.y)
    w = getattr(dataset, "w", None)
    ids = getattr(dataset, "ids", None)

    if y.ndim == 1:
        y_task = y
    else:
        y_task = y[:, task_index]

    if w is None:
        w_task = None
    else:
        w = np.asarray(w)
        if w.ndim == 1:
            w_task = w
        else:
            w_task = w[:, task_index]

    if ids is not None:
        ids = np.asarray(ids)

    return dc.data.NumpyDataset(X, y_task, w=w_task, ids=ids)


def filter_missing_labels(dataset: dc.data.Dataset) -> dc.data.NumpyDataset:
    w = getattr(dataset, "w", None)
    if w is None:
        return dataset
    w = np.asarray(w)
    if w.ndim != 1:
        raise ValueError("Expected single-task dataset with 1D weights.")
    mask = w != 0
    X = np.asarray(dataset.X)[mask]
    y = np.asarray(dataset.y)[mask]
    w = w[mask]
    ids = getattr(dataset, "ids", None)
    if ids is not None:
        ids = np.asarray(ids)[mask]
    return dc.data.NumpyDataset(X, y, w=w, ids=ids)


def slice_features(dataset: dc.data.Dataset, n_features: Optional[int]) -> dc.data.NumpyDataset:
    X = np.asarray(dataset.X)
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("Non-numeric features detected. Use a numeric featurizer such as 'rdkit' or 'ecfp'.")
    if n_features is not None and n_features > 0:
        if X.ndim != 2:
            raise ValueError("Feature slicing expects a 2D feature matrix.")
        X = X[:, :n_features]
    return dc.data.NumpyDataset(X, np.asarray(dataset.y), w=getattr(dataset, "w", None), ids=getattr(dataset, "ids", None))


def subsample(dataset: dc.data.Dataset, max_samples: Optional[int], seed: int) -> dc.data.NumpyDataset:
    if max_samples is None or max_samples <= 0:
        return dataset
    n = len(dataset)
    if n <= max_samples:
        return dataset
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    X = np.asarray(dataset.X)[idx]
    y = np.asarray(dataset.y)[idx]
    w = getattr(dataset, "w", None)
    if w is not None:
        w = np.asarray(w)[idx]
    ids = getattr(dataset, "ids", None)
    if ids is not None:
        ids = np.asarray(ids)[idx]
    return dc.data.NumpyDataset(X, y, w=w, ids=ids)


def apply_transformers(
    dataset: dc.data.Dataset,
    transformers: List[dc.trans.Transformer],
) -> dc.data.Dataset:
    out = dataset
    for transformer in transformers:
        out = transformer.transform(out)
    return out


def impute_features_median(
    train: dc.data.Dataset,
    valid: dc.data.Dataset,
    test: dc.data.Dataset,
) -> Tuple[dc.data.NumpyDataset, dc.data.NumpyDataset, dc.data.NumpyDataset]:
    X_train_raw = np.asarray(train.X)
    if X_train_raw.shape[0] == 0:
        raise ValueError("Training split is empty after preprocessing; cannot fit imputer or train models.")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)

    def transform_or_empty(split: dc.data.Dataset) -> np.ndarray:
        X_split = np.asarray(split.X)
        if X_split.shape[0] == 0:
            return np.empty((0, X_train.shape[1]), dtype=X_train.dtype)
        return imputer.transform(X_split)

    X_valid = transform_or_empty(valid)
    X_test = transform_or_empty(test)

    train_out = dc.data.NumpyDataset(
        X_train,
        np.asarray(train.y),
        w=getattr(train, "w", None),
        ids=getattr(train, "ids", None),
    )
    valid_out = dc.data.NumpyDataset(
        X_valid,
        np.asarray(valid.y),
        w=getattr(valid, "w", None),
        ids=getattr(valid, "ids", None),
    )
    test_out = dc.data.NumpyDataset(
        X_test,
        np.asarray(test.y),
        w=getattr(test, "w", None),
        ids=getattr(test, "ids", None),
    )
    return train_out, valid_out, test_out


def build_metrics() -> List[dc.metrics.Metric]:
    return [
        dc.metrics.Metric(dc.metrics.r2_score, np.mean),
        dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean),
        dc.metrics.Metric(dc.metrics.mae_score, np.mean),
        dc.metrics.Metric(dc.metrics.rms_score, np.mean),
    ]


def evaluate_model(model: dc.models.Model, datasets: Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset],
                   metrics: List[dc.metrics.Metric], transformers: List[dc.trans.Transformer]) -> Dict[str, Dict[str, float]]:
    train, valid, test = datasets

    def eval_or_empty(split: dc.data.Dataset) -> Dict[str, float]:
        if len(split) == 0:
            return {}
        return model.evaluate(split, metrics, transformers)

    return {
        "train": eval_or_empty(train),
        "valid": eval_or_empty(valid),
        "test": eval_or_empty(test),
    }


def run_pysr_subprocess(
    train: dc.data.Dataset,
    valid: dc.data.Dataset,
    test: dc.data.Dataset,
    args,
    sr_ops: Optional[List[str]],
) -> Dict[str, object]:
    if shutil.which("julia") is None:
        return {
            "available": False,
            "error": "Julia executable not found on PATH; install Julia to enable PySR.",
            "train_seconds": None,
            "best_expression": None,
            "scores": None,
        }
    if int(args.sr_population_size) <= 1:
        return {
            "available": False,
            "error": "PySR requires --sr-population-size > 1.",
            "train_seconds": None,
            "best_expression": None,
            "scores": None,
        }

    binary_ops, unary_ops = split_ops_for_pysr(sr_ops)
    with tempfile.TemporaryDirectory(prefix="pysr_bench_") as tmpdir:
        data_path = f"{tmpdir}/data.npz"
        out_path = f"{tmpdir}/out.json"
        np.savez_compressed(
            data_path,
            X_train=np.asarray(train.X),
            y_train=np.asarray(train.y).reshape(-1),
            X_valid=np.asarray(valid.X),
            y_valid=np.asarray(valid.y).reshape(-1),
            X_test=np.asarray(test.X),
            y_test=np.asarray(test.y).reshape(-1),
        )
        config = {
            "niterations": int(args.sr_iterations),
            "ncycles_per_iteration": int(args.sr_cycles_per_iteration),
            "populations": int(args.sr_populations),
            "population_size": int(args.sr_population_size),
            "tournament_selection_n": max(1, min(15, int(args.sr_population_size) - 1)),
            "topn": max(1, min(12, int(args.sr_population_size))),
            "maxsize": int(args.sr_maxsize),
            "maxdepth": int(args.sr_maxdepth),
            "parsimony": float(args.sr_parsimony_penalty),
            "optimizer_probability": float(args.sr_optimizer_probability),
            "optimizer_nrestarts": int(args.sr_opt_restarts),
            "seed": int(args.seed),
            "binary_ops": binary_ops,
            "unary_ops": unary_ops,
            "model_selection": str(args.sr_model_selection),
            "data_path": data_path,
            "out_path": out_path,
        }
        cfg_path = f"{tmpdir}/config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        code = r"""
import inspect
import json
import time
import numpy as np
from scipy import stats
from pysr import PySRRegressor

def pearson_r2(y_true, y_pred):
    try:
        return float(stats.pearsonr(y_true, y_pred)[0]**2)
    except Exception:
        return float("nan")

def regression_scores(y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rms = float(np.sqrt(np.mean(err**2)))
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - y_mean)**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else float("nan")
    return {
        "mean-r2_score": r2,
        "mean-pearson_r2_score": pearson_r2(y_true, y_pred),
        "mean-mae_score": mae,
        "mean-rms_score": rms,
    }

def scores_for_split(model, X, y):
    if X.shape[0] == 0:
        return {}
    pred = model.predict(X)
    return regression_scores(y, pred)

with open(__import__("sys").argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
arr = np.load(cfg["data_path"])
X_train = arr["X_train"]
y_train = arr["y_train"]
X_valid = arr["X_valid"]
y_valid = arr["y_valid"]
X_test = arr["X_test"]
y_test = arr["y_test"]

sig = inspect.signature(PySRRegressor.__init__)
params = sig.parameters

pysr_kwargs = dict(
    niterations=cfg["niterations"],
    populations=cfg["populations"],
    population_size=cfg["population_size"],
    tournament_selection_n=cfg["tournament_selection_n"],
    topn=cfg["topn"],
    maxsize=cfg["maxsize"],
    maxdepth=cfg["maxdepth"],
    binary_operators=cfg["binary_ops"],
    unary_operators=cfg["unary_ops"],
    model_selection=cfg["model_selection"],
    batching=False,
    progress=False,
    verbosity=0,
    random_state=cfg["seed"],
)

if "ncycles_per_iteration" in params:
    pysr_kwargs["ncycles_per_iteration"] = cfg["ncycles_per_iteration"]
elif "ncyclesperiteration" in params:
    pysr_kwargs["ncyclesperiteration"] = cfg["ncycles_per_iteration"]

if "parsimony" in params:
    pysr_kwargs["parsimony"] = cfg["parsimony"]

if "optimizer_probability" in params:
    pysr_kwargs["optimizer_probability"] = cfg["optimizer_probability"]
elif "optimize_probability" in params:
    pysr_kwargs["optimize_probability"] = cfg["optimizer_probability"]

if "optimizer_nrestarts" in params:
    pysr_kwargs["optimizer_nrestarts"] = cfg["optimizer_nrestarts"]
elif "optimize_nrestarts" in params:
    pysr_kwargs["optimize_nrestarts"] = cfg["optimizer_nrestarts"]

model = PySRRegressor(**pysr_kwargs)
t0 = time.time()
model.fit(X_train, y_train)
train_seconds = time.time() - t0

try:
    best_expression = str(model.get_best())
except Exception:
    best_expression = None

out = {
    "available": True,
    "error": None,
    "train_seconds": train_seconds,
    "best_expression": best_expression,
    "scores": {
        "train": scores_for_split(model, X_train, y_train),
        "valid": scores_for_split(model, X_valid, y_valid),
        "test": scores_for_split(model, X_test, y_test),
    },
}
with open(cfg["out_path"], "w", encoding="utf-8") as f:
    json.dump(out, f)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code, cfg_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or f"PySR subprocess failed (code {proc.returncode})."
            return {
                "available": False,
                "error": msg,
                "train_seconds": None,
                "best_expression": None,
                "scores": None,
            }
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_with_compatible_kwargs(
    loader: Loader,
    *,
    featurizer: object,
    splitter: str,
) -> Tuple[List[str], Tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset], List[dc.trans.Transformer]]:
    params = inspect.signature(loader).parameters
    kwargs = {}

    if "featurizer" in params:
        kwargs["featurizer"] = featurizer
    if "splitter" in params:
        kwargs["splitter"] = splitter
    elif "split" in params:
        kwargs["split"] = splitter
    if "transformers" in params:
        kwargs["transformers"] = []
    if "reload" in params:
        kwargs["reload"] = True

    return loader(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare symbolic regression models on MoleculeNet regression datasets.")
    parser.add_argument("--dataset", default="delaney", choices=sorted(REGRESSION_LOADERS.keys()))
    parser.add_argument("--featurizer", default="rdkit", help="Numeric featurizer recommended for symbolic regression (e.g., rdkit, ecfp).")
    parser.add_argument("--splitter", default="random", help="Splitter name for MoleculeNet loader.")
    parser.add_argument("--task-index", type=int, default=0, help="Task index to use if dataset is multitask.")
    parser.add_argument("--n-features", type=int, default=0, help="If >0, slice the first N features for both models.")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--normalize-x", action="store_true", help="Apply NormalizationTransformer to features.")
    parser.add_argument("--normalize-y", action="store_true", help="Apply NormalizationTransformer to labels.")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--sr-iterations", type=int, default=500)
    parser.add_argument("--sr-population-size", type=int, default=50)
    parser.add_argument("--sr-populations", type=int, default=1)
    parser.add_argument("--sr-cycles-per-iteration", type=int, default=40)
    parser.add_argument("--sr-maxsize", type=int, default=12)
    parser.add_argument("--sr-maxdepth", type=int, default=12)
    parser.add_argument("--sr-tree-depth", type=int, default=3)
    parser.add_argument("--sr-parsimony-penalty", type=float, default=0.0)
    parser.add_argument("--sr-optimizer-probability", type=float, default=0.5)
    parser.add_argument("--sr-opt-steps", type=int, default=500)
    parser.add_argument("--sr-opt-lr", type=float, default=0.1)
    parser.add_argument("--sr-opt-restarts", type=int, default=2)
    parser.add_argument("--sr-ops", type=str, default="add,sub,mul,div,neg,sin,cos,log,exp")
    parser.add_argument(
        "--sr-model-selection",
        type=str,
        default="best",
        choices=["best", "pareto_score", "min_cost"],
        help="Equation selection strategy for both DeepChem SR and PySR.",
    )

    parser.add_argument("--out", type=str, default="", help="Optional path to write JSON results.")
    args = parser.parse_args()

    loader = REGRESSION_LOADERS[args.dataset]
    featurizer = args.featurizer
    _, (train, valid, test), _ = load_with_compatible_kwargs(
        loader,
        featurizer=featurizer,
        splitter=args.splitter,
    )

    train = select_single_task(train, args.task_index)
    valid = select_single_task(valid, args.task_index)
    test = select_single_task(test, args.task_index)

    train = filter_missing_labels(train)
    valid = filter_missing_labels(valid)
    test = filter_missing_labels(test)

    n_features = args.n_features if args.n_features and args.n_features > 0 else None
    train = slice_features(train, n_features)
    valid = slice_features(valid, n_features)
    test = slice_features(test, n_features)

    train, valid, test = impute_features_median(train, valid, test)

    train = subsample(train, args.max_train if args.max_train > 0 else None, args.seed)
    valid = subsample(valid, args.max_valid if args.max_valid > 0 else None, args.seed + 1)
    test = subsample(test, args.max_test if args.max_test > 0 else None, args.seed + 2)

    transformers: List[dc.trans.Transformer] = []
    if args.normalize_x:
        transformers.append(dc.trans.NormalizationTransformer(transform_X=True, dataset=train))
    if args.normalize_y:
        transformers.append(dc.trans.NormalizationTransformer(transform_y=True, dataset=train))

    if transformers:
        train = apply_transformers(train, transformers)
        valid = apply_transformers(valid, transformers)
        test = apply_transformers(test, transformers)

    metrics = build_metrics()

    sr_ops = parse_csv_list(args.sr_ops)
    sr_topn = max(1, min(12, int(args.sr_population_size)))
    sr_deepchem = SymbolicRegressionModel(
        ops=sr_ops,
        niterations=args.sr_iterations,
        population_size=args.sr_population_size,
        populations=args.sr_populations,
        ncycles_per_iteration=args.sr_cycles_per_iteration,
        maxsize=args.sr_maxsize,
        maxdepth=args.sr_maxdepth,
        tree_depth=args.sr_tree_depth,
        parsimony_penalty=args.sr_parsimony_penalty,
        optimizer_probability=args.sr_optimizer_probability,
        opt_steps=args.sr_opt_steps,
        opt_lr=args.sr_opt_lr,
        opt_nrestarts=args.sr_opt_restarts,
        seed=args.seed,
        options_kwargs={
            "model_selection": args.sr_model_selection,
            "topn": sr_topn,
        },
        use_multiprocessing=False,
    )

    sr_deepchem.fit(train)

    sr_deepchem_scores = evaluate_model(sr_deepchem, (train, valid, test), metrics, transformers)
    sr_pysr_result = run_pysr_subprocess(train, valid, test, args, sr_ops)

    results = {
        "dataset": args.dataset,
        "featurized": featurizer,
        "split_sizes": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        },
        "scores": {
            "symbolic_regression_deepchem": sr_deepchem_scores,
            "symbolic_regression_pysr": sr_pysr_result["scores"] if sr_pysr_result["available"] else {},
        },
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
