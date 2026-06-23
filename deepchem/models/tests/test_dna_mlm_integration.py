import torch
from deepchem.feat.sequence_tokenizer import DNACharTokenizer
from deepchem.models.torch_models.dna_mlm_collator import DNAMLMDataCollator
from deepchem.models.torch_models.dna_mlm_model import DNATransformerForMLM


def test_dna_mlm_full_pipeline():
    # Step 1: Tokenize
    tokenizer = DNACharTokenizer(max_length=8)
    features = tokenizer.featurize(["ACGTACGT", "ACGT"])

    input_ids = torch.tensor(features["input_ids"])
    attention_mask = torch.tensor(features["attention_mask"])

    # Step 2: Apply dynamic masking
    collator = DNAMLMDataCollator(
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=len(tokenizer.vocab),
        pad_token_id=tokenizer.pad_id,
    )

    batch = {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
    }

    masked_batch = collator(batch)

    # Step 3: Forward through model
    model = DNATransformerForMLM()

    outputs = model.model(
        input_ids=masked_batch["input_ids"],
        attention_mask=masked_batch["attention_mask"],
        labels=masked_batch["labels"],
    )

    # Check loss exists
    assert outputs.loss is not None

    # Check logits shape
    assert outputs.logits.shape[0] == 2
