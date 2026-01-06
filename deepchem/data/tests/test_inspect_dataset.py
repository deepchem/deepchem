import pandas as pd

from deepchem.data.inspect_dataset import inspect_dataframe


def test_inspect_dataframe_basic():
    """Basic sanity check for inspect_dataframe on a small toy dataset."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, None],
        "label": [1, 0, 1, None],
        "smiles": ["CCO", "C", "CCCC", None],
    })

    summary = inspect_dataframe(df, n_sample=2)

    # High-level counts.
    assert summary["n_examples"] == 4
    assert summary["n_columns"] == 3

    # Task column detection.
    assert "label" in summary["task_columns"]

    # Check missing counts reported.
    col_info = {c["name"]: c for c in summary["columns"]}
    assert col_info["a"]["n_missing"] == 1
    assert col_info["smiles"]["n_missing"] == 1

    # Samples list length should be 2.
    assert len(summary["samples"]) == 2
    # Samples should be dicts with the expected keys.
    assert all(isinstance(s, dict) for s in summary["samples"])
    assert all(
        {"a", "label", "smiles"} <= set(s.keys()) for s in summary["samples"])
