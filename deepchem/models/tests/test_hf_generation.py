import os

import pytest
import torch

@pytest.fixture
def hf_tokenizer(tmpdir):
    filepath = os.path.join(tmpdir, 'smiles.txt')
    with open(filepath, 'w') as f:
        f.write(
            'CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1\nCC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1\n'
        )
        f.write(
            'COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC\nOCCn1cc(CNc2cccc3c2CCCC3)nn1\n'
        )
        f.write(
            'CCCCCCc1ccc(C#Cc2ccc(C#CC3=CC=C(CCC)CC3)c(C3CCCCC3)c2)c(F)c1\nO=C(NCc1ccc(F)cc1)N1CC=C(c2c[nH]c3ccccc23)CC1\n'
        )

    from tokenizers import ByteLevelBPETokenizer
    from transformers.models.roberta import RobertaTokenizerFast

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=filepath,
                    vocab_size=1_000,
                    min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    tokenizer_path = os.path.join(tmpdir, 'tokenizer')
    os.makedirs(tokenizer_path)
    tokenizer.save_model(tokenizer_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


@pytest.mark.hf
def test_hf_generation(hf_tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel
    from deepchem.models.torch_models import HuggingFaceModel

    config = GPT2Config(
        vocab_size=hf_tokenizer.vocab_size,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)

    hf_model = HuggingFaceModel(
        model=model,
        tokenizer=hf_tokenizer,
        task="generation",
        device=torch.device("cpu"),
    )

    hf_model._ensure_built()

    outputs = hf_model.generate(["Hello world"], max_new_tokens=5)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)