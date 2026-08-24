"""Utilities for honest model + featurizer benchmarking on small molecular datasets.

Small QSAR datasets (a few hundred to a few thousand molecules) are easy to overfit:
random cross-validation is typically optimistic on novel-chemistry test sets, and a
finely-tuned ensemble selected on a small validation split often *loses* to a single
robust model on a truly-held-out series. ``benchmark_featurizer_model_combinations``
runs a combinatorial featurizer x model search under **scaffold** cross-validation (the
honest default) and surfaces these pitfalls as warnings.

Example
-------
>>> import deepchem as dc
>>> from deepchem.utils.benchmark_utils import benchmark_featurizer_model_combinations
>>> results, insights = benchmark_featurizer_model_combinations(
...     smiles=train_smiles, y=train_y,
...     featurizers=["ecfp", "rdkit_desc"],
...     models=["ridge", "rf"], splitter="scaffold", k=5)
>>> print(results.sort_values("rae").head())
>>> for line in insights: print(line)
"""
from typing import List, Sequence, Tuple, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _rae(y: np.ndarray, p: np.ndarray) -> float:
    """Relative Absolute Error = sum|y-p| / sum|y-median(y)| (<1 beats median baseline)."""
    denom = np.abs(y - np.median(y)).sum()
    return float(np.abs(y - p).sum() / denom) if denom > 0 else float("nan")


def _get_featurizer(name: str):
    import deepchem as dc
    name = name.lower()
    if name in ("ecfp", "morgan", "circular"):
        return dc.feat.CircularFingerprint(size=2048, radius=2)
    if name in ("rdkit_desc", "rdkit", "descriptors"):
        return dc.feat.RDKitDescriptors()
    if name in ("maccs", "maccskeys"):
        return dc.feat.MACCSKeysFingerprint()
    if name in ("mol2vec", "pubchem"):
        return dc.feat.PubChemFingerprint()
    raise ValueError(f"Unknown featurizer '{name}'. "
                     "Supported: ecfp, rdkit_desc, maccs, pubchem.")


def _get_model(name: str):
    """Return an (estimator, needs_scaling) tuple of a scikit-learn regressor."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    name = name.lower()
    table = {
        "ridge": (Ridge(alpha=1.0), True),
        "rf": (RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0), False),
        "extratrees": (ExtraTreesRegressor(n_estimators=400, n_jobs=-1, random_state=0), False),
        "histgbm": (HistGradientBoostingRegressor(random_state=0), False),
        "knn": (KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1), True),
        "svr": (SVR(C=10.0), True),
    }
    if name not in table:
        raise ValueError(f"Unknown model '{name}'. Supported: {sorted(table)}")
    return table[name]


def _make_splits(smiles: Sequence[str], y: np.ndarray, splitter: str, k: int, seed: int):
    """Return a list of (train_idx, val_idx) folds using a DeepChem splitter."""
    import deepchem as dc
    n = len(smiles)
    ids = np.arange(n)
    dataset = dc.data.NumpyDataset(X=ids.reshape(-1, 1), y=y, ids=np.asarray(smiles, dtype=object))
    if splitter == "scaffold":
        sp = dc.splits.ScaffoldSplitter()
    elif splitter in ("butina", "cluster"):
        sp = dc.splits.ButinaSplitter()
    elif splitter == "random":
        sp = dc.splits.RandomSplitter()
    else:
        raise ValueError("splitter must be 'scaffold', 'butina' or 'random'")
    folds = sp.k_fold_split(dataset, k)
    out = []
    for train_ds, val_ds in folds:
        out.append((train_ds.X.ravel().astype(int), val_ds.X.ravel().astype(int)))
    return out


def benchmark_featurizer_model_combinations(
        smiles: Sequence[str],
        y: Sequence[float],
        featurizers: Sequence[str] = ("ecfp", "rdkit_desc"),
        models: Sequence[str] = ("ridge", "rf", "histgbm", "knn"),
        splitter: str = "scaffold",
        k: int = 5,
        seed: int = 0,
) -> Tuple["object", List[str]]:
    """Benchmark every featurizer x model combination under honest cross-validation.

    Parameters
    ----------
    smiles: Sequence[str]
        Training SMILES strings.
    y: Sequence[float]
        Regression targets aligned to ``smiles``.
    featurizers: Sequence[str]
        Names of DeepChem featurizers to try. Supported: ``ecfp``, ``rdkit_desc``,
        ``maccs``, ``pubchem``.
    models: Sequence[str]
        scikit-learn regressor names. Supported: ``ridge``, ``rf``, ``extratrees``,
        ``histgbm``, ``knn``, ``svr``.
    splitter: str
        Cross-validation splitter: ``scaffold`` (default, honest for novel chemistry),
        ``butina`` (Tanimoto cluster) or ``random``.
    k: int
        Number of CV folds.
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    results: pandas.DataFrame
        One row per (featurizer, model) with out-of-fold ``rae``, ``mae``, ``r2`` and,
        for reference, the same metrics under a random split (``random_rae``) so you can
        see how optimistic random CV is.
    insights: List[str]
        Human-readable warnings/recommendations, e.g. small-N caution and random-CV
        optimism.

    Notes
    -----
    On small datasets, prefer the single most robust configuration over a finely-tuned
    stack: tuned ensembles tend to overfit the validation split and lose on a
    series-shifted test set. The returned insights flag this automatically.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    smiles = list(smiles)
    y = np.asarray(y, dtype=float)
    n = len(y)
    folds = _make_splits(smiles, y, splitter, k, seed)
    folds_random = _make_splits(smiles, y, "random", k, seed)

    # cache featurizations once
    feats: Dict[str, np.ndarray] = {}
    for fname in featurizers:
        try:
            X = _get_featurizer(fname).featurize(smiles)
            X = np.nan_to_num(np.asarray(X, dtype=float))
            feats[fname] = X
        except Exception as exc:  # pragma: no cover - featurizer edge cases
            logger.warning("Skipping featurizer %s: %s", fname, exc)

    def _oof(fold_list, X, model_name):
        est, needs_scale = _get_model(model_name)
        oof = np.full(n, np.nan)
        for tr, va in fold_list:
            Xt, Xv = X[tr], X[va]
            if needs_scale:
                sc = StandardScaler().fit(Xt)
                Xt, Xv = sc.transform(Xt), sc.transform(Xv)
            from sklearn.base import clone
            m = clone(est).fit(Xt, y[tr])
            oof[va] = m.predict(Xv)
        return oof

    rows = []
    for fname, X in feats.items():
        for mname in models:
            try:
                oof = _oof(folds, X, mname)
                oof_r = _oof(folds_random, X, mname)
            except Exception as exc:  # pragma: no cover
                logger.warning("Skipping %s+%s: %s", fname, mname, exc)
                continue
            rows.append({
                "featurizer": fname, "model": mname,
                "rae": round(_rae(y, oof), 4),
                "mae": round(float(mean_absolute_error(y, oof)), 4),
                "r2": round(float(r2_score(y, oof)), 4),
                "random_rae": round(_rae(y, oof_r), 4),
            })
    results = pd.DataFrame(rows).sort_values("rae").reset_index(drop=True)

    insights: List[str] = []
    if n < 500:
        insights.append(
            f"[small-N: {n} molecules] Prefer the single most robust model over a "
            "finely-tuned ensemble; tuned stacks overfit small validation splits and "
            "lose on series-shifted test sets.")
    if len(results):
        gap = (results["rae"] - results["random_rae"]).max()
        if gap > 0.05:
            insights.append(
                f"[optimism] random CV is ~{gap:.2f} RAE more optimistic than "
                f"{splitter} CV - the data likely spans novel chemistry; trust the "
                f"{splitter}-CV numbers.")
        best = results.iloc[0]
        insights.append(
            f"Best config: {best['featurizer']} + {best['model']} "
            f"({splitter}-CV RAE {best['rae']}, MAE {best['mae']}, R2 {best['r2']}).")
    return results, insights
