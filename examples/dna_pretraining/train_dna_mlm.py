import torch
from torch.utils.data import DataLoader

from deepchem.feat.sequence_tokenizer import DNACharTokenizer
from deepchem.models.torch_models.dna_mlm_model import DNATransformerForMLM
from deepchem.models.torch_models.dna_mlm_collator import DNAMLMDataCollator

from fasta_dataset import FASTADataset


def train():

    fasta_file = "sample_sequences.fasta"

    dataset = FASTADataset(fasta_file, seq_length=64)

    tokenizer = DNACharTokenizer(max_length=64)

    features = tokenizer.featurize(dataset.sequences)

    input_ids = torch.tensor(features["input_ids"])
    attention_mask = torch.tensor(features["attention_mask"])

    dataloader = DataLoader(
        list(zip(input_ids, attention_mask)),
        batch_size=4,
        shuffle=True
    )

    collator = DNAMLMDataCollator(
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=len(tokenizer.vocab),
        pad_token_id=tokenizer.pad_id,
        mask_probability=0.15
    )

    model = DNATransformerForMLM(
        vocab_size=len(tokenizer.vocab)
    )

    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=1e-4
    )

    for epoch in range(3):

        for batch in dataloader:

            batch_input_ids, batch_attention_mask = batch

            batch_dict = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask
            }

            inputs = collator(batch_dict)

            outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch:", epoch, "Loss:", loss.item())


if __name__ == "__main__":
    train()