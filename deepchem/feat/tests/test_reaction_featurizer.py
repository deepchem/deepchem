from deepchem.feat.reaction_featurizer import RxnFeaturizer
import pytest
import os

@pytest.mark.torch
def test_featurize():
    """Test that RxnFeaturizer.featurize() correctly featurizes the reactions,
    correctly outputs the input_ids and attention_mask.
    """
    from deepchem.feat.reaction_featurizer import RxnFeaturizer
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    featurizer = RxnFeaturizer(tokenizer, sep_reagent=True)
    reaction = ['CCS(=O)(=O)Cl.OCCBr>CCN(CC)CC.CCOCC>CCS(=O)(=O)OCCBr']
    feats = featurizer.featurize(reaction)
    assert (feats.shape == (1, 2, 2, 1))
    
