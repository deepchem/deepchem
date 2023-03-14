from deepchem.feat.huggingface_featurizer import HuggingFaceFeaturizer


def testHuggingFaceFeaturizer():
    # NOTE: The test depends on the sanity of the pretrained tokenizer,
    # which is not guaranteed since the external resource can be modified.
    from transformers import RobertaTokenizerFast
    hf_tokenizer = RobertaTokenizerFast.from_pretrained(
        "seyonec/PubChem10M_SMILES_BPE_60k")

    featurizer = HuggingFaceFeaturizer(tokenizer=hf_tokenizer)
    output = featurizer.featurize(['CC(=O)C', 'CC'])
    assert len(output) == 2
    assert output[0]['input_ids'] == [0, 262, 263, 51, 13, 39, 2]
    assert output[0]['attention_mask'] == [1, 1, 1, 1, 1, 1, 1]

    assert output[1]['input_ids'] == [0, 262, 2]
    assert output[1]['attention_mask'] == [1, 1, 1]
