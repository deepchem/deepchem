import torch

from deepchem.feat.sequence_tokenizer import DNACharTokenizer
from deepchem.models.torch_models.dna_mlm_model import DNATransformerForMLM
from deepchem.models.torch_models.dna_mlm_collator import DNAMLMDataCollator


def test_dna_pretraining_forward():

    sequences = [
        "ACGTACGTACGT",
        "TTGACCGTACGA"
    ]

    tokenizer = DNACharTokenizer(max_length=32)

    features = tokenizer.featurize(sequences)

    input_ids = torch.tensor(features["input_ids"])
    attention_mask = torch.tensor(features["attention_mask"])

    collator = DNAMLMDataCollator(
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=len(tokenizer.vocab),
        pad_token_id=tokenizer.pad_id,
        mask_probability=0.15
    )

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    inputs = collator(batch)

    model = DNATransformerForMLM(
        vocab_size=len(tokenizer.vocab)
    )

    outputs = model.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["labels"]
    )

    loss = outputs.loss

    assert loss is not None