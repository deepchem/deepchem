"""Small CLI and library helpers to inspect tabular datasets.

Usage:
    python -m deepchem.data.inspect_dataset path/to/data.csv --n-sample 5

The module focuses on lightweight, dependency-minimal inspection using pandas.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".csv") or path.endswith(".txt"):
        return pd.read_csv(path)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    # For SDF or other formats, users can pass a DataFrame directly.
    raise ValueError("Unsupported file type. Pass a CSV/TSV or a DataFrame.")


def inspect_dataframe(df: pd.DataFrame, n_sample: int = 5) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["n_examples"] = int(len(df))
    summary["n_columns"] = int(len(df.columns))
    summary["columns"] = []

    for col in df.columns:
        col_ser = df[col]
        col_info: Dict[str, Any] = {"name": col, "dtype": str(col_ser.dtype)}
        col_info["n_missing"] = int(col_ser.isna().sum())
        if pd.api.types.is_numeric_dtype(col_ser):
            col_info["mean"] = None if col_ser.dropna().empty else float(col_ser.mean())
            col_info["std"] = None if col_ser.dropna().empty else float(col_ser.std())
            col_info["min"] = None if col_ser.dropna().empty else float(col_ser.min())
            col_info["max"] = None if col_ser.dropna().empty else float(col_ser.max())
        else:
            top = col_ser.dropna().astype(str).value_counts().head(5).to_dict()
            col_info["top_values"] = {k: int(v) for k, v in top.items()}
            # If column looks like SMILES, include length stats
            if col.lower() in ("smiles", "smile"):
                lengths = col_ser.dropna().astype(str).map(len)
                if not lengths.empty:
                    col_info["smiles_len_mean"] = float(lengths.mean())
                    col_info["smiles_len_std"] = float(lengths.std())
        summary["columns"].append(col_info)

    # Simple task/label heuristics: look for columns named 'label' or starting with 'task'
    task_cols = [c for c in df.columns if c.lower().startswith("task") or c.lower() == "label"]
    summary["task_columns"] = task_cols

    # Basic class balance for task columns when possible
    balances = {}
    for tc in task_cols:
        ser = df[tc]
        counts = ser.dropna().value_counts().to_dict()
        balances[tc] = {str(k): int(v) for k, v in counts.items()}
    summary["task_balances"] = balances

    # Samples
    try:
        sample_df = df.sample(min(n_sample, len(df)), random_state=0)
        # Convert to plain Python types
        summary["samples"] = json.loads(sample_df.to_json(orient="records"))
    except Exception:
        summary["samples"] = []

    return summary


def print_summary(summary: Dict[str, Any], out=sys.stdout) -> None:
    json.dump(summary, out, indent=2)
    out.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect tabular datasets (CSV/TSV)")
    p.add_argument("path", help="Path to CSV/TSV file")
    p.add_argument("--n-sample", type=int, default=5, help="Number of example rows to show")
    return p


def main(argv=None) -> int:
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
