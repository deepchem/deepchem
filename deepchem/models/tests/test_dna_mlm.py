import torch
from deepchem.models.torch_models.dna_mlm_collator import DNAMLMDataCollator


def test_mlm_collator_shapes():
    batch_size = 2
    seq_len = 8
    vocab_size = 10

    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    collator = DNAMLMDataCollator(
        mask_token_id=1,
        vocab_size=vocab_size,
        pad_token_id=0,
    )

    batch = {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
    }

    output = collator(batch)

    assert output["input_ids"].shape == (batch_size, seq_len)
    assert output["labels"].shape == (batch_size, seq_len)


def test_no_padding_masked():
    input_ids = torch.tensor([[0, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    collator = DNAMLMDataCollator(
        mask_token_id=1,
        vocab_size=10,
        pad_token_id=0,
    )

    batch = {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
    }

    output = collator(batch)

    assert output["labels"][0, 0] == -100
