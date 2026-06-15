import pytest


@pytest.mark.torch
def test_genomic_call():
    """Test __call__ method for the featurizer, which is inherited from HuggingFace's PreTrainedTokenizerFast"""
    from deepchem.feat.dnabert_tokenizer import DNABertFeaturizer
    sequences = ["ATGCGTACGTTAGCTAGC", "GGCTAACCGTATCGGATC"]
    long_sequence = [
        "ATGCGTACGTTAGCTAGCATGCGTACGTTAGCTAGCATGCGTACGTTAGCTAGCATGCGTACGTTAGCTAGC"
    ]
    featurizer = DNABertFeaturizer.from_pretrained("zhihan1996/DNABERT-2-117M",
                                                   trust_remote_code=True)
    embedding = featurizer(sequences, add_special_tokens=True, truncation=True)
    embedding_long = featurizer(long_sequence * 2,
                                add_special_tokens=True,
                                truncation=True)
    for emb in [embedding, embedding_long]:
        print(list(emb.keys()))
        print(f"The length of input ids is {len(emb['input_ids'])}")
        print(f"The length of input ids is {len(emb['attention_mask'])}")
        print(emb.keys())
        print("\n")


def test_genomic_featurize():
    """Test the .featurize method, which will convert the dictionary output to an array

    Checks that all sequences are featurized and that each featurization
    contains input_ids and attention_mask
    """
    from deepchem.feat.dnabert_tokenizer import DNABertFeaturizer
    sequences = ["ATGCGTACGTTAGCTAGC", "GGCTAACCGTATCGGATC"]
    long_sequence = [
        "ATGCGTACGTTAGCTAGCATGCGTACGTTAGCTAGCATGCGTACGTTAGCTAGCATGCGTACGTTAGCTAGC"
    ]
    featurizer = DNABertFeaturizer.from_pretrained("zhihan1996/DNABERT-2-117M",
                                                   trust_remote_code=True)
    max_length = 100
    feat_kwargs = {
        'add_special_tokens': True,
        'truncation': True,
        'padding': 'max_length',
        'max_length': max_length
    }
    feats = featurizer.featurize(sequences, **feat_kwargs)
    print(f"The length of the features are {len(feats)}")
    print([len(f) for f in feats])
    print([len(f[0]) for f in feats])
    print(f"\n The Inputs ids for the first sequence are: \n {feats[0][0]}")
    print(
        f"\n The Attention Masks for the first sequence are: \n {feats[0][1]}")
    print(f"\n The Inputs ids for the second sequence are: \n {feats[1][0]}")
    print(
        f"\n The Attention Masks for the second sequence are: \n {feats[1][1]}")

    long_feat = featurizer.featurize(long_sequence, **feat_kwargs)

    print(f"\n The length of long sequence are {len(long_feat)}")
    print([len(f) for f in long_feat])
    print([len(f[0]) for f in long_feat])
    print(f"\n The Inputs ids for the first sequence are: \n {long_feat[0][0]}")
    print(
        f"\n The Attention Masks for the first sequence are: \n {long_feat[0][1]}"
    )
