import pytest
import numpy as np
import deepchem as dc


@pytest.fixture(scope="session")
def pretrained_checkpoint_dir(tmp_path_factory):
    """Download HF weights once and save as a local DeepChem checkpoint."""
    from deepchem.models.torch_models.dnabert import Dnabert
    checkpoint_dir = tmp_path_factory.mktemp("dnabert_pretrained")
    tokenizer_path = 'IronHead44/DNABERT-2-117M'
    model = Dnabert(task='mlm',
                    tokenizer_path=tokenizer_path,
                    model_dir=str(checkpoint_dir))
    model.load_from_pretrained(tokenizer_path, from_hf_checkpoint=True)
    model.save_checkpoint()
    return str(checkpoint_dir)


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
