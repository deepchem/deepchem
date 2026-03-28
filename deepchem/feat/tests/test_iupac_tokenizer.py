from deepchem.feat.iupac_tokenizer import IUPACTokenizer


def test_simple_iupac():
    tokenizer = IUPACTokenizer()
    tokens = tokenizer.featurize(["2-methylpropane"])
    assert list(tokens[0]) == ['2', '-', 'methylpropane']