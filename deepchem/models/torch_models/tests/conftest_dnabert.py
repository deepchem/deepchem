import pytest
import numpy as np
import deepchem as dc


@pytest.fixture
def genomic_regression_dataset():
    sequences = [
        "ATGCGTACGTTAGCTAGCATGCGTACG",
        "GGCTAACCGTATCGGATCAAGTCCTAG",
        "TTAAGCCGTACGATCGATCGATCGATCG",
        "CCGATCGATCGATCGATCGATCGATCGA",
        "ATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "AATTCCGGAATTCCGGAATTCCGGAATT",
        "CCGGTTAACCGGTTAACCGGTTAACCGG",
        "TTAGGCCAATTAGGCCAATTAGGCCAAT",
        "GGCCAATTGGCCAATTGGCCAATTGGCC",
    ]
    np.random.seed(42)
    y = np.random.rand(10, 1)
    dataset = dc.data.NumpyDataset(X=np.array(sequences), y=y)
    return dataset


@pytest.fixture
def genomic_multitask_regression_dataset():
    sequences = [
        "ATGCGTACGTTAGCTAGCATGCGTACG",
        "GGCTAACCGTATCGGATCAAGTCCTAG",
        "TTAAGCCGTACGATCGATCGATCGATCG",
        "CCGATCGATCGATCGATCGATCGATCGA",
        "ATCGATCGATCGATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "AATTCCGGAATTCCGGAATTCCGGAATT",
        "CCGGTTAACCGGTTAACCGGTTAACCGG",
        "TTAGGCCAATTAGGCCAATTAGGCCAAT",
        "GGCCAATTGGCCAATTGGCCAATTGGCC",
    ]
    np.random.seed(42)
    y = np.random.rand(10, 2)
    dataset = dc.data.NumpyDataset(X=np.array(sequences), y=y)
    return dataset
