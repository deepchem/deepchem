"""Tests for deepchem.utils.benchmark_utils."""
import numpy as np
import pytest


@pytest.mark.torch  # only needs sklearn+rdkit, but grouped with light integration tests
def test_benchmark_featurizer_model_combinations():
    """Smoke-test the honest benchmark on a tiny synthetic dataset."""
    pytest.importorskip("sklearn")
    pytest.importorskip("rdkit")
    from deepchem.utils.benchmark_utils import benchmark_featurizer_model_combinations

    smiles = [
        "c1ccccc1", "CCO", "CCN", "CCC(=O)O", "c1ccncc1", "CC(C)C", "CCCCCC",
        "c1ccc(O)cc1", "CC(=O)Nc1ccccc1", "OCC(O)CO", "c1ccc2ccccc2c1", "CN1CCCC1",
        "CCOCC", "O=C(O)c1ccccc1", "Cc1ccccc1C", "CCCCO", "NCCO", "c1ccc(Cl)cc1",
        "OCCO", "c1ccc(N)cc1", "CCCCCCCC", "CC(C)Cc1ccccc1", "COc1ccccc1", "CCCCN",
    ] * 2
    from rdkit import Chem
    y = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() + 0.1 * (i % 3)
                  for i, s in enumerate(smiles)], dtype=float)

    results, insights = benchmark_featurizer_model_combinations(
        smiles=smiles, y=y, featurizers=["ecfp", "rdkit_desc"],
        models=["ridge", "rf"], splitter="scaffold", k=3)

    assert len(results) > 0
    for col in ("featurizer", "model", "rae", "mae", "r2", "random_rae"):
        assert col in results.columns
    # results are sorted best-first and RAE is finite
    assert np.isfinite(results.iloc[0]["rae"])
    # small-N insight should fire (48 molecules)
    assert any("small-N" in s for s in insights)


@pytest.mark.torch
def test_benchmark_random_splitter():
    pytest.importorskip("sklearn")
    pytest.importorskip("rdkit")
    from deepchem.utils.benchmark_utils import benchmark_featurizer_model_combinations
    smiles = ["C" * (i % 6 + 1) for i in range(40)] + ["c1ccccc1", "CCO"] * 5
    y = np.arange(len(smiles), dtype=float) + np.random.RandomState(0).randn(len(smiles))
    results, _ = benchmark_featurizer_model_combinations(
        smiles=smiles, y=y, featurizers=["ecfp"], models=["ridge"],
        splitter="random", k=3)
    assert len(results) == 1
