"""Small CLI and library helpers to inspect tabular datasets.

Usage (CLI)
-----------
Run from the command line with:

.. code-block:: bash

    python -m deepchem.data.inspect_dataset path/to/data.csv --n-sample 5

This module focuses on lightweight, dependency-minimal inspection using
pandas and is intended to provide a quick overview of tabular datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Sequence

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    """Load a tabular dataset from disk into a DataFrame.

    Parameters
    ----------
    path : str
        Path to a CSV, TXT (comma-separated), or TSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded table.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = str(path)
    if path.endswith(".csv") or path.endswith(".txt"):
        return pd.read_csv(path)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    raise ValueError("Unsupported file type. Expected a CSV, TXT, or TSV path.")


def inspect_dataframe(df: pd.DataFrame, n_sample: int = 5) -> Dict[str, Any]:
    """Compute a lightweight summary for a tabular dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to inspect.
    n_sample : int, optional (default 5)
        Number of example rows to include in the summary. The actual
        number of sampled rows is ``min(n_sample, len(df))``.

    Returns
    -------
    dict
        A JSON-serializable summary with the following keys:

        - ``"n_examples"`` : int, number of rows
        - ``"n_columns"`` : int, number of columns
        - ``"columns"`` : list of per-column statistics
        - ``"task_columns"`` : list of column names that look like labels/tasks
        - ``"task_balances"`` : dict mapping task columns to value counts
        - ``"samples"`` : list of sampled rows as plain Python dicts
    """
    summary: Dict[str, Any] = {}
    summary["n_examples"] = int(len(df))
    summary["n_columns"] = int(len(df.columns))
    summary["columns"] = []

    for col in df.columns:
        col_ser = df[col]
        col_info: Dict[str, Any] = {"name": col, "dtype": str(col_ser.dtype)}
        col_info["n_missing"] = int(col_ser.isna().sum())

        if pd.api.types.is_numeric_dtype(col_ser):
            non_missing = col_ser.dropna()
            if non_missing.empty:
                col_info["mean"] = None
                col_info["std"] = None
                col_info["min"] = None
                col_info["max"] = None
            else:
                col_info["mean"] = float(non_missing.mean())
                col_info["std"] = float(non_missing.std())
                col_info["min"] = float(non_missing.min())
                col_info["max"] = float(non_missing.max())
        else:
            top = col_ser.dropna().astype(str).value_counts().head(5).to_dict()
            col_info["top_values"] = {k: int(v) for k, v in top.items()}

            # If column looks like SMILES, include length statistics.
            if col.lower() in ("smiles", "smile"):
                lengths = col_ser.dropna().astype(str).map(len)
                if not lengths.empty:
                    col_info["smiles_len_mean"] = float(lengths.mean())
                    col_info["smiles_len_std"] = float(lengths.std())

        summary["columns"].append(col_info)

    # Simple task/label heuristics: look for columns named 'label' or starting with 'task'.
    task_cols = [
        c for c in df.columns
        if c.lower().startswith("task") or c.lower() == "label"
    ]
    summary["task_columns"] = task_cols

    # Basic class balance for task columns when possible.
    balances: Dict[str, Dict[str, int]] = {}
    for tc in task_cols:
        ser = df[tc]
        counts = ser.dropna().value_counts().to_dict()
        balances[tc] = {str(k): int(v) for k, v in counts.items()}
    summary["task_balances"] = balances

    # Sample example rows and convert them to plain Python types.
    try:
        sample_df = df.sample(min(n_sample, len(df)), random_state=0)
        summary["samples"] = json.loads(sample_df.to_json(orient="records"))
    except Exception:
        summary["samples"] = []

    return summary


def print_summary(summary: Dict[str, Any], out=sys.stdout) -> None:
    """Print a JSON summary to a file-like object.

    Parameters
    ----------
    summary : dict
        Summary dictionary produced by :func:`inspect_dataframe`.
    out : file-like, optional
        Stream to write JSON to. Defaults to ``sys.stdout``.
    """
    json.dump(summary, out, indent=2)
    out.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the inspect_dataset CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect tabular datasets (CSV/TSV)")
    parser.add_argument("path", help="Path to CSV/TSV file")
    parser.add_argument(
        "--n-sample",
        type=int,
        default=5,
        help="Number of example rows to show",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the inspect_dataset CLI.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments to parse. If ``None``, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code compatible with :func:`sys.exit`. Returns 0 on success
        and 2 if the dataset cannot be loaded.
    """
    args = _build_parser().parse_args(argv)
    try:
        df = load_table(args.path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 2

    summary = inspect_dataframe(df, n_sample=args.n_sample)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
