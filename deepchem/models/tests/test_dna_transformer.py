import torch
from deepchem.models.torch_models import DNATransformer


def test_dna_transformer_forward():
    model = DNATransformer(num_labels=2)

    batch_size = 2
    seq_length = 8

    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    assert outputs.logits.shape == (batch_size, 2)
